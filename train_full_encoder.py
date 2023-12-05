import os
import random
import argparse
from pathlib import Path
import copy
import logging

import itertools
import time
from types import MethodType
import yaml

import torch
import torch.nn.functional as F
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import (
    AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel, StableDiffusionControlNetPipeline,
    DDIMScheduler, StableDiffusionPipeline
    )
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from transformers import CLIPVisionModel
# from custom_clip.modeling_clip import CLIPVisionModel

from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available, generate
from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
# if is_torch2_available():
#     from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
# else:
#     from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
from packaging import version

from con_net.dataset.dataset2 import LaionHumanSD
from con_net.metric import Metric
from con_net.utils import copy_src, image_grid

logger = get_logger(__name__)
logger.setLevel(logging.INFO)


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
    }
    

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, image_encoder):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        self.image_encoder = image_encoder

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, batch, accelerator):
        # # whether to train the whole image encoder
        image_embeds = self.image_encoder(batch["clip_images"].to(accelerator.device)).last_hidden_state

        image_embeds_ = []
        for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
            if drop_image_embed == 1:
                image_embeds_.append(torch.zeros_like(image_embed))
            else:
                image_embeds_.append(image_embed)
        image_embeds = torch.stack(image_embeds_)


        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual and compute loss
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default='config/base.yaml',
    )
    parser.add_argument(
        "--ip_ckpt",
        type=str,
        default='IP-Adapter/models/ip-adapter_sd15.bin',
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # merge yaml cfg to args
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    # TODO: check argparse.Namespace 的使用方法
    # args = argparse.Namespace(**cfg, **vars(args))
    for k, v in cfg.items():
        setattr(args, k, v)

    # TODO: parse yaml with OmegaConf
    args.learning_rate = float(args.learning_rate)
    args.num_train_steps = int(float(args.num_train_steps))

    dt_string = time.strftime("%Y-%m-%d-%H-%M")
    # dt_string = time.strftime("%Y-%m-%d-%H-%M-%S")  # 不同进程的同步问题
    args.output_dir = os.path.join(args.output_dir, dt_string)

    return args


def build_inferecne_pipe(args):
    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    vae_model_path = "/mnt/petrelfs/majie/model_checkpoint/sd-vae-ft-mse"
    infer_vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        scheduler=ddim_scheduler,
        vae=infer_vae,
        feature_extractor=None,
        safety_checker=None
    )
    pipe.__call__ = MethodType(generate, pipe)
    return pipe



def validate(ip_adapter, args, step, metric, accelerator, validate_data):
    pipe = build_inferecne_pipe(args)

    # build ip-adapter for inference
    from ip_adapter import IPAdapter as IPInference
    ip_adapter_inference = IPInference(sd_pipe=pipe, image_encoder_path=args.image_encoder_path, 
                                       ip_ckpt=args.ip_ckpt, device=accelerator.device)

    # Load trained image proj model and attention processors
    ip_adapter_inference.image_proj_model.load_state_dict(ip_adapter.module.image_proj_model.state_dict())

    ip_inference_layers = torch.nn.ModuleList(ip_adapter_inference.pipe.unet.attn_processors.values())
    ip_inference_layers.load_state_dict(ip_adapter.module.adapter_modules.state_dict())
    
    sim = []
    # TODO: 参考图片通过clip编码，在批量处理时应该预先处理
    for i in range(len(validate_data.data)):
        # if i > 5:
        #     break
        j = random.randint(0, len(validate_data.data) - 1)
        each = validate_data.data[j]

        key = each['key']
        try:
            clip_image = Image.open(each['img_path']).convert("RGB")
        except Exception:
            continue
        prompt = each['prompt']

        # whether to use general prompt
        # prompt = 'best quality, high quality, simple background, full body, standing,'

        width, height = clip_image.size
        width = (width // 8) * 8
        height = (height // 8) * 8
        clip_image = clip_image.resize((width, height), Image.BILINEAR)

        # generate
        images = ip_adapter_inference.generate(pil_image=clip_image, prompt=prompt, width=width, height=height,
                                               num_samples=4, num_inference_steps=50)

        # save
        results = [clip_image] + images
        grid = image_grid(results, 1, len(results))
        save_dir = f'{args.output_dir}/{step}'
        os.makedirs(save_dir, exist_ok=True)
        grid.save(f'{save_dir}/{key}.png')

        # evaluate
        sim.append(metric.clip_sim(images, clip_image))

        break

    if len(sim) != 0 and accelerator.is_main_process:
        sim = sum(sim) / len(sim)
        logger.info(f'Step: {step}, sim: {sim}')

        logs = {f'valid/clip_sim': sim}
        accelerator.log(logs, step=step)
    
    del pipe, ip_adapter_inference, ip_inference_layers
    torch.cuda.empty_cache()

    return sim if type(sim) == float else 0


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    # print(f'rank, {accelerator.process_index}')
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        copy_src(os.path.join(args.output_dir, "src"))

        file_handler = logging.FileHandler(os.path.join(args.output_dir, "log.txt"))
        logger.logger.addHandler(file_handler)  

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers("ip-adapter", config=tracker_config)

    # Load Metric
    metric = Metric(device=accelerator.device)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModel.from_pretrained(args.image_encoder_path)
    # TODO: 使用ViT-H有256个patch，而使用ViT-L有49个patch，check

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.vision_model.post_layernorm.requires_grad_(False)

    # build pipeline for inference
    # pipe = build_inferecne_pipe(args)
    
    # ip-adapter
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.hidden_size,
    )

    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, image_encoder)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # TODO: 官网的教程说不需要.to(device)的操作
    # ##unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    # image_encoder.to(accelerator.device)
    # image_encoder.to(accelerator.device, dtype=weight_dtype)

    # optimizer
    params_to_opt = itertools.chain(ip_adapter.adapter_modules.parameters(), ip_adapter.image_proj_model.parameters(), ip_adapter.image_encoder.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    # dataloader
    train_dataset = LaionHumanSD(json_file=args.json_file, tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    if accelerator.is_main_process:
        # batches = len(train_dataloader) / accelerator.num_processes
        logger.info(f"Loaded {len(train_dataset)} train samples, {len(train_dataloader) / accelerator.num_processes} batches")

    # lr scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.num_train_steps * accelerator.num_processes
    )

    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        ip_adapter, optimizer, train_dataloader, lr_scheduler)

    if args.resume is not None:
        accelerator.load_state(args.resume)
        begin_step = int(args.resume.split('-')[-1])
    else:
        begin_step = 0

    train_dataloader_iter = iter(train_dataloader)
    for step in range(begin_step, args.num_train_steps):
        try:
            batch = next(train_dataloader_iter)
        except StopIteration:
            train_dataloader_iter = iter(train_dataloader)
            batch = next(train_dataloader_iter)
        
        with accelerator.accumulate(ip_adapter):
            # Convert images to latent space
            with torch.no_grad():
                latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]
            
            noise_pred = ip_adapter(noisy_latents, timesteps, encoder_hidden_states, batch, accelerator)
    
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
            
            # Backpropagate
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # for name, param in ip_adapter.named_parameters():
            #     if param.grad is None:
            #         print(name, param.requires_grad)
            # exit(0)

            logs = {"train/loss": avg_loss, 'train/lr': lr_scheduler.get_last_lr()[0]}
            accelerator.log(logs, step=step)
            
            if step % args.log_steps == 0:
                # if accelerator.process_index != 0:
                #     sim = validate(ip_adapter, args, step, metric, accelerator, train_dataset)
                if accelerator.is_main_process:
                    logger.info("Current time: {}, Step {}, lr, {}, step_loss: {}".format(
                        time.strftime("%Y-%m-%d-%H-%M-%S"), step, lr_scheduler.get_last_lr()[0], avg_loss))
        
        if step % args.save_steps == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
            accelerator.save_state(save_path)
        

if __name__ == "__main__":
    main()    
