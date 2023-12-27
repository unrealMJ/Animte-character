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
    DDIMScheduler, StableDiffusionPipeline, UNetMotionModel, MotionAdapter
    )
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPVisionModel

from con_net.dataset.dataset2 import BaseDataset, BaseVideoDataset
from con_net.metric import Metric
from con_net.utils import copy_src, image_grid
from omegaconf import OmegaConf
import einops


logger = get_logger(__name__)
logger.setLevel(logging.INFO)


def collate_fn(data):
    videos = torch.stack([example["video"] for example in data])  # [b, f, c, h, w]
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    references = torch.stack([example["reference"] for example in data])
    control_videos = torch.stack([example["control_video"] for example in data])

    return {
        "videos": videos,
        "text_input_ids": text_input_ids,
        "references": references,
        "control_videos": control_videos,
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

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    reference_net = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # load reference weight
    state_dict = torch.load(args.reference_path, map_location='cpu')
    reference_net.load_state_dict(state_dict)

    motion_adapter = MotionAdapter.from_pretrained(args.motion_path)
    unet = UNetMotionModel.from_unet2d(unet, motion_adapter)

    if args.control_type == 'canny':
        controlnet = ControlNetModel.from_pretrained(args.canny_controlnet)
    elif args.control_type == 'pose':
        controlnet = ControlNetModel.from_pretrained(args.pose_controlnet)
    else:
        raise Exception('control_type must be canny or pose')

    # freeze parameters of models to save more memory
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)
    reference_net.requires_grad_(False)

    for k, v in unet.named_parameters():
        if 'motion_modules' not in k:
            v.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # TODO: 官网的教程说不需要.to(device)的操作
    # ##unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)
    reference_net.to(accelerator.device, dtype=weight_dtype)

    # optimizer
    params_to_opt = []
    for k, v in unet.named_parameters():
        if v.requires_grad:
            params_to_opt.append(v)
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    # dataloader
    train_dataset = BaseVideoDataset(json_file=args.json_file, tokenizer=tokenizer, control_type=args.control_type)
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
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler)

    if args.resume is not None:
        accelerator.load_state(args.resume)
        begin_step = int(args.resume.split('-')[-1])
    else:
        begin_step = 0

    # register reference_net hook
    to_k_hook = []
    to_v_hook = []

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

    # register unet hook
    def to_k2_forward(module, input_, output_):
        # output_: [b*f, h*w, c]
        # tmp: [b, h*w, c] b=1
        tmp = to_k_hook.pop()
        tmp = tmp[:, None]  # [b, 1, h*w, c]
        tmp = tmp.repeat(1, args.num_frames, 1, 1)  # [b, f, h*w, c]
        tmp = einops.rearrange(tmp, 'b f l c -> (b f) l c')

        assert output_.shape == tmp.shape, f'{output_.shape}, {tmp.shape}'
        res = torch.cat([output_, tmp], dim=1)
        return res

    def to_v2_forward(module, input_, output_):
        # output_: [b*f, h*w, c]
        tmp = to_v_hook.pop()
        tmp = tmp[:, None]  # [b, 1, h*w, c]
        tmp = tmp.repeat(1, args.num_frames, 1, 1)  # [b, f, h*w, c]
        tmp = einops.rearrange(tmp, 'b f l c -> (b f) l c')
        assert output_.shape == tmp.shape, f'{output_.shape}, {tmp.shape}'
        res = torch.cat([output_, tmp], dim=1)
        return res
    
    for k, v in unet.named_modules():
        # attn1是self-attn, attn2是cross-attn
        # 这里注册的顺序不对，需要重新排序
        if 'attn1.to_k' in k and 'motion_modules' not in k:
            print(f'register hook for {k}, {v}')
            v.register_forward_hook(to_k2_forward)
        if 'attn1.to_v' in k and 'motion_modules' not in k:
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
        
        # if accelerator.is_main_process:
        #     print(f'batch["videos"] {batch["videos"].shape}')
        #     print(f'batch["text_input_ids"] {batch["text_input_ids"].shape}')
        #     print(f'batch["references"] {batch["references"].shape}')
        #     print(f'batch["control_videos"] {batch["control_videos"].shape}')

        video_length = batch["videos"].shape[1]
        with accelerator.accumulate(reference_net):
            # Convert images to latent space
            with torch.no_grad():
                videos = batch["videos"].to(accelerator.device, dtype=weight_dtype)
                
                videos = einops.rearrange(videos, 'b f c h w -> (b f) c h w')
                latents = vae.encode(videos).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = einops.rearrange(latents, '(b f) c h w -> b c f h w', f=video_length)
                
                # reference image
                reference_image = batch["references"].to(accelerator.device,dtype=weight_dtype)
                reference_latents = vae.encode(reference_image).latent_dist.sample()
                reference_latents = reference_latents * vae.config.scaling_factor  # [b, c, h, w] b=1

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)  # [b, c, f, h, w]
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)  # [b, c, f, h, w]
        
            with torch.no_grad():
                encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]
            
            control_image = batch["control_videos"].to(dtype=weight_dtype)
            control_image = einops.rearrange(control_image, 'b f c h w -> (b f) c h w')

            # print(f'control_image {control_image.shape}')
            # print(f'noisy_latents {einops.rearrange(noisy_latents, "b c f h w -> (b f) c h w").shape}')

            # timesteps的shape是（bs），表示为每个图片加不同程度的噪声
            # video的bs是1，frame是24，所以timesteps需要扩展到24
            control_timesteps = timesteps[:, None].repeat(1, video_length).reshape(-1)
            control_encoder_hidden_states = encoder_hidden_states[:, None].repeat(1, video_length, 1, 1)
            control_encoder_hidden_states = einops.rearrange(control_encoder_hidden_states, 'b f l c -> (b f) l c')
            # print(f'control_timesteps {control_timesteps.shape}')
            # print(f'control_encoder_hidden_states {control_encoder_hidden_states.shape}')
            # exit(0)
            # controlnet
            # TODO: 检查control_timesteps
            down_block_res_samples, mid_block_res_sample = controlnet(
                einops.rearrange(noisy_latents, 'b c f h w -> (b f) c h w'),
                control_timesteps,
                encoder_hidden_states=control_encoder_hidden_states,
                controlnet_cond=control_image,
                return_dict=False,
            )
            # TODO: 检查controlnet的结果是否需要reshape成video格式

            # Predict the noise residual
            tmp_timesteps = torch.zeros_like(timesteps, device=latents.device).long()
            _ = reference_net(
                reference_latents,
                tmp_timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            to_k_hook.reverse()
            to_v_hook.reverse()
            
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
            
            # reset hook list !!!!
            assert len(to_k_hook) == len(to_v_hook)
            assert len(to_k_hook) == 0
            to_k_hook.clear()
            to_v_hook.clear()

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
