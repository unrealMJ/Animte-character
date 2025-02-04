import os
import random
import argparse
from pathlib import Path
import copy
import logging

import itertools
import time
from types import MethodType

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
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPVisionModel

from con_net.dataset.dataset2 import LaionHumanSD, CCTVDataset, TikTokDataset, BaseDataset
from con_net.metric import Metric
from con_net.utils import copy_src, image_grid
from omegaconf import OmegaConf


logger = get_logger(__name__)
logger.setLevel(logging.INFO)


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    references = torch.stack([example["reference"] for example in data])
    control_image = torch.stack([example["control_image"] for example in data])

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "references": references,
        "control_image": control_image,
    }
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default='config/train/cctv.yaml',
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
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1, 
        help="For distributed training: local_rank"
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    cfg = OmegaConf.load(args.config)

    # TODO: check argparse.Namespace 的使用方法
    # args = argparse.Namespace(**cfg, **vars(args))
    for k, v in cfg.items():
        setattr(args, k, v)

    dt_string = time.strftime("%Y-%m-%d-%H-%M")
    # dt_string = time.strftime("%Y-%m-%d-%H-%M-%S")  # 不同进程的同步问题
    args.output_dir = os.path.join(args.output_dir, dt_string)

    return args


def build_inferecne_pipe(args, appearence_controlnet, pose_controlnet):
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
    
    appearence_controlnet = appearence_controlnet.to(dtype=torch.float16)
    controlnet = MultiControlNetModel([appearence_controlnet, pose_controlnet])

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        scheduler=ddim_scheduler,
        vae=infer_vae,
        feature_extractor=None,
        safety_checker=None
    )
    return pipe


def validate(args, appearence_controlnet, pose_controlnet, step, metric, accelerator, validate_data):
    appearence_controlnet = accelerator.unwrap_model(appearence_controlnet)
    pipe = build_inferecne_pipe(args, appearence_controlnet, pose_controlnet)

    sim = []
    # TODO: 参考图片通过clip编码，在批量处理时应该预先处理
    for i in range(len(validate_data.data)):
        # if i > 5:
        #     break
        j = random.randint(0, len(validate_data.data) - 1)

        item = validate_data.data[j]
        
        role = item['role']
        image = Image.open(item['image']).convert("RGB")
        reference = Image.open(item['reference']).convert("RGB")
        pose = Image.open(item['pose']).convert("RGB")
        prompt = 'best quality'

        # whether to use general prompt
        # prompt = 'best quality, high quality, simple background, full body, standing,'

        width, height = reference.size
        width = (width // 8) * 8
        height = (height // 8) * 8

        reference = reference.resize((width, height), Image.BILINEAR)
        pose = pose.resize((width, height), Image.BILINEAR)
        
        # generate
        results = pipe(prompt=prompt, width=width, height=height, num_inference_steps=50, image=[reference, pose], num_images_per_prompt=4).images

        # ## evaluate
        # sim.append(metric.clip_sim(images, clip_image))

        # save
        results = [reference] + [pose] + results
        grid = image_grid(results, 1, len(results))

        save_dir = f'{args.output_dir}/{step}'
        os.makedirs(save_dir, exist_ok=True)

        current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
        grid.save(f'{save_dir}/{role}_{current_time}.png')

        break

    if len(sim) != 0 and accelerator.is_main_process:
        sim = sum(sim) / len(sim)
        logger.info(f'Step: {step}, sim: {sim}')

        logs = {f'valid/clip_sim': sim}
        accelerator.log(logs, step=step)
    
    del pipe
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
        tracker_config.pop('json_file') # remove list type
        accelerator.init_trackers("controlnet", config=tracker_config)

    # Load Metric
    # metric = Metric(device=accelerator.device)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    
    apperence_controlnet = ControlNetModel.from_unet(unet)
    if args.control_type == 'canny':
        controlnet = ControlNetModel.from_pretrained(args.canny_controlnet)
    elif args.control_type == 'pose':
        controlnet = ControlNetModel.from_pretrained(args.pose_controlnet)
    else:
        raise Exception('control_type must be canny or pose')

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # TODO: 官网的教程说不需要.to(device)的操作
    # ##unet.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    multi_controlnet = MultiControlNetModel([apperence_controlnet, controlnet])
    # optimizer
    params_to_opt = apperence_controlnet.parameters()
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    # dataloader
    train_dataset = BaseDataset(json_file=args.json_file, tokenizer=tokenizer, control_type=args.control_type)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    if accelerator.is_main_process:
        logger.info(f"Loaded {len(train_dataset)} train samples, {len(train_dataloader) / accelerator.num_processes} batches")

    # lr scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.num_train_steps * accelerator.num_processes
    )

    # Prepare everything with our `accelerator`.
    apperence_controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        apperence_controlnet, optimizer, train_dataloader, lr_scheduler)

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
        
        with accelerator.accumulate(apperence_controlnet):
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
            
            reference_image = batch["references"].to(dtype=weight_dtype)
            control_image = batch["control_image"].to(dtype=weight_dtype)

            down_block_res_samples, mid_block_res_sample = multi_controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=[reference_image, control_image],
                conditioning_scale=[1, 1],
                return_dict=False,
            )

            # Predict the noise residual
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=[
                    sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
            ).sample

            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
            
            # Backpropagate
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            logs = {"train/loss": avg_loss, 'train/lr': lr_scheduler.get_last_lr()[0]}
            accelerator.log(logs, step=step)
            
            if step % args.log_steps == 0:
                if accelerator.is_main_process:
                    logger.info("Current time: {}, Step {}, lr, {}, step_loss: {}".format(
                        time.strftime("%Y-%m-%d-%H-%M-%S"), step, lr_scheduler.get_last_lr()[0], avg_loss))
        
        if step % args.save_steps == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
            accelerator.save_state(save_path)
        

if __name__ == "__main__":
    main()    
