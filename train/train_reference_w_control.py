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

from con_net.dataset.dataset_laionhuman_w_control import LaionHumanSD, CCTVDataset, TikTokDataset, BaseDataset
from con_net.metric import Metric
from con_net.utils import copy_src, image_grid
from omegaconf import OmegaConf
from inference.validate_wo_control import Inference
import torchvision.transforms as transforms
import einops
from controlnet_aux import HEDdetector
import numpy as np
import cv2

logger = get_logger(__name__)
logger.setLevel(logging.INFO)


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    references = torch.stack([example["reference"] for example in data])
    control_images = torch.stack([example["control_image"] for example in data])

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "references": references,
        "control_images": control_images,
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


def validate_save(grid, prompt, exp_dir, step):
    os.makedirs(f'{exp_dir}/{step}', exist_ok=True)
    prompt = prompt.replace('/', '')[:100]
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    grid.save(f'{exp_dir}/{step}/{prompt}_{current_time}.png')


def validate(args, pipeline, reference_net, step, to_k_hook, to_v_hook, control_type):
    val_image_lines = open('data/validate_images.txt').readlines()
    val_prompt_lines = open('data/validate_prompts.txt').readlines()

    if control_type == 'hed':
        hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')

    for i in range(len(val_image_lines)):
        image = val_image_lines[i].strip()
        prompt = val_prompt_lines[i].strip()

        reference_image = Image.open(image).convert("RGB")

        width = args.validate_size
        height = args.validate_size
        reference = reference_image.resize((width, height), Image.BILINEAR)
        raw_image = reference

        reference = transforms.ToTensor()(reference)
        reference = transforms.Normalize([0.5], [0.5])(reference)
        reference = reference.unsqueeze(0)

        reference = reference.to('cuda').to(dtype=torch.float16)
        reference_latents = pipeline.vae.encode(reference).latent_dist.sample()
        reference_latents = reference_latents * pipeline.vae.config.scaling_factor

        text_input_ids = pipeline.tokenizer(
            prompt,
            max_length=pipeline.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        text_input_ids = text_input_ids[None].to('cuda')

        encoder_hidden_states = pipeline.text_encoder(text_input_ids)[0]
        
        timesteps = torch.tensor([0]).long().to('cuda')
        _ = reference_net(reference_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

        for i in range(len(to_k_hook)):
            feature = to_k_hook[i]
            feature = einops.repeat(feature, 'b l c -> (b n) l c', n=8)
            to_k_hook[i] = feature
        
        for i in range(len(to_v_hook)):
            feature = to_v_hook[i]
            feature = einops.repeat(feature, 'b l c -> (b n) l c', n=8)
            to_v_hook[i] = feature

        if control_type == 'canny':
            control_image = np.array(raw_image)
            control_image = cv2.Canny(control_image, 100, 200)
            control_image = control_image[:, :, None]
            control_image = np.concatenate([control_image, control_image, control_image], axis=2)
            control_image = Image.fromarray(control_image)
        elif control_type == 'hed':
            control_image = hed(image)

        control_image = control_image.resize((width, height), Image.BILINEAR)
        results = pipeline(prompt=prompt, width=width, height=height, image=control_image, num_inference_steps=50, num_images_per_prompt=4).images

        # reset hook list !!!!
        assert len(to_k_hook) == len(to_v_hook)
        global k_idx, v_idx
        k_idx = 0
        v_idx = 0
        to_k_hook.clear()
        to_v_hook.clear()

        all_images = [reference_image.resize((width, height))] + [control_image] + results
        grid = image_grid(all_images, 1, len(all_images))

        validate_save(grid, prompt, os.path.join(args.output_dir, 'validate'), step)
        

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
    reference_net = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    controlnet = None
    if args.use_control:
        if args.control_type == 'canny':
            controlnet = ControlNetModel.from_pretrained(args.canny_controlnet)
        elif args.control_type == 'hed':
            controlnet = ControlNetModel.from_pretrained(args.hed_controlnet)
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

    # optimizer
    params_to_opt = reference_net.parameters()
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    # dataloader
    train_dataset = LaionHumanSD(data_root=args.data_root, json_file=args.json_file, tokenizer=tokenizer, use_control=args.use_control, control_type=args.control_type)
    # train_dataset = BaseDataset(json_file=args.json_file, tokenizer=tokenizer, control_type=args.control_type)
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
    reference_net, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        reference_net, optimizer, train_dataloader, lr_scheduler)

    if args.resume is not None:
        accelerator.load_state(args.resume)
        begin_step = int(args.resume.split('-')[-1])
    else:
        begin_step = 0

    # register reference_net hook
    to_k_hook = []
    to_v_hook = []
    global k_idx, v_idx
    k_idx = 0
    v_idx = 0

    def to_k_forward(module, input_, output_):
        to_k_hook.append(output_)

    def to_v_forward(module, input_, output_):
        to_v_hook.append(output_)
    
    for k, v in reference_net.named_modules():
        # attn1是self-attn, attn2是cross-attn
        # 这里注册的顺序不对，需要重新排序
        # 似乎不需要排序，forward放入list的tensor是顺序的
        if 'attn1.to_k' in k:
            print(f'register hook for {k}, {v}')
            v.register_forward_hook(to_k_forward)
        if 'attn1.to_v' in k:
            print(f'register hook for {k}, {v}')
            v.register_forward_hook(to_v_forward)

    def to_k2_forward(module, input_, output_):
        global k_idx
        if k_idx > len(to_k_hook) - 1:
            k_idx = 0
        tmp = to_k_hook[k_idx]
        k_idx += 1
        assert output_.shape == tmp.shape, f'ouput_ shape {output_.shape} tmp shape {tmp.shape}'
        res = torch.cat([output_, tmp], dim=1)
        return res

    def to_v2_forward(module, input_, output_):
        # output_: [b, hw, c]
        global v_idx
        if v_idx > len(to_v_hook) - 1:
            v_idx = 0
        tmp = to_v_hook[v_idx]
        v_idx += 1
        assert output_.shape == tmp.shape, f'ouput_ shape {output_.shape} tmp shape {tmp.shape}'
        res = torch.cat([output_, tmp], dim=1)
        return res

    for k, v in unet.named_modules():
        # attn1是self-attn, attn2是cross-attn
        # 这里注册的顺序不对，需要重新排序
        if 'attn1.to_k' in k:
            print(f'register hook for {k}, {v}')
            v.register_forward_hook(to_k2_forward)
        if 'attn1.to_v' in k:
            print(f'register hook for {k}, {v}')
            v.register_forward_hook(to_v2_forward)

    # Train!
    train_dataloader_iter = iter(train_dataloader)
    for step in range(begin_step, args.num_train_steps):
        try:
            batch = next(train_dataloader_iter)
        except StopIteration:
            train_dataloader_iter = iter(train_dataloader)
            batch = next(train_dataloader_iter)
        
        with accelerator.accumulate(reference_net):
            # Convert images to latent space
            with torch.no_grad():
                latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # reference image
                reference_image = batch["references"].to(accelerator.device,dtype=weight_dtype)
                reference_latents = vae.encode(reference_image).latent_dist.sample()
                reference_latents = reference_latents * vae.config.scaling_factor

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

            control_image = batch["control_images"].to(dtype=weight_dtype)
            down_block_res_samples, mid_block_res_sample = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=control_image,
                return_dict=False,
            )
            # Predict the noise residual
            zero_timesteps = torch.zeros_like(timesteps, device=latents.device).long()

            aux = reference_net(
                reference_latents,
                zero_timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=[
                    sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
            ).sample

            # 避免unused parameters
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean") + 0 * aux.sum()

            # loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
            
            # Backpropagate
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # reset hook list !!!!
            assert len(to_k_hook) == len(to_v_hook), f'to_k_hook len {len(to_k_hook)}, to_v_hook len {len(to_v_hook)}'
            k_idx = 0
            v_idx = 0
            to_k_hook.clear()
            to_v_hook.clear()

            logs = {"train/loss": avg_loss, 'train/lr': lr_scheduler.get_last_lr()[0]}
            accelerator.log(logs, step=step)

            if step % args.log_steps == 0:
                if accelerator.is_main_process:
                    logger.info("Current time: {}, Step {}, lr, {}, step_loss: {}".format(
                        time.strftime("%Y-%m-%d-%H-%M-%S"), step, lr_scheduler.get_last_lr()[0], avg_loss))

        if step > 0 and step % args.save_steps == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
            accelerator.save_state(save_path)

        if accelerator.is_main_process and step % args.validate_steps == 0:
            pipeline = StableDiffusionControlNetPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                controlnet=controlnet,
                scheduler=noise_scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
            validate(args, pipeline, reference_net, step, to_k_hook, to_v_hook, args.control_type)


if __name__ == "__main__":
    main()    
