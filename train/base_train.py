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


class BaseTrainer:
    def __init__(self):
        self.cfg = self.parse_args()
        self.logger = get_logger(__name__)
        self.logger.setLevel(logging.INFO)
        
        pass
    
    def parse_args(self):
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

    def collate_fn(self, data):
        images = torch.stack([example["image"] for example in data])
        # reference_prompt = torch.cat([example["reference_prompt"] for example in data], dim=0)
        text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
        references = torch.stack([example["reference"] for example in data])

        return {
            "images": images,
            "text_input_ids": text_input_ids,
            "references": references,
        }
        
    def init_context(self):
        logging_dir = Path(self.cfg.output_dir, self.cfg.logging_dir)

        accelerator_project_config = ProjectConfiguration(project_dir=self.cfg.output_dir, logging_dir=logging_dir)
        accelerator = Accelerator(
            mixed_precision=self.cfg.mixed_precision,
            log_with=self.cfg.report_to,
            project_config=accelerator_project_config,
        )
        if accelerator.is_main_process:
            if self.cfg.output_dir is not None:
                os.makedirs(self.cfg.output_dir, exist_ok=True)
            copy_src(os.path.join(self.cfg.output_dir, "src"))

            file_handler = logging.FileHandler(os.path.join(self.cfg.output_dir, "log.txt"))
            self.logger.logger.addHandler(file_handler)  

        if accelerator.is_main_process:
            tracker_config = vars(copy.deepcopy(self.cfg))
            tracker_config.pop('json_file') # remove list type
            accelerator.init_trackers("controlnet", config=tracker_config)

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.accelerator = accelerator
        self.weight_dtype = weight_dtype  # torch.float16
        
        # if self.accelerator.is_main_process:
        # print(f'weight dtype: {self.weight_dtype}')
        self.logger.info(f"weight dtype: {self.weight_dtype}")

    def train(self):
        train_dataloader_iter = iter(self.train_dataloader)
        for step in range(self.begin_step, self.end_step):
            try:
                batch = next(train_dataloader_iter)
            except StopIteration:
                train_dataloader_iter = iter(self.train_dataloader)
                batch = next(train_dataloader_iter)
            
            with self.accelerator.accumulate(self.reference_net):
                # Convert images to latent space
                with torch.no_grad():
                    latents = self.vae.encode(batch["images"].to(self.accelerator.device, dtype=self.weight_dtype)).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                    
                    # reference image
                    reference_image = batch["references"].to(self.accelerator.device,dtype=self.weight_dtype)
                    reference_latents = self.vae.encode(reference_image).latent_dist.sample()
                    reference_latents = reference_latents * self.vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            
                with torch.no_grad():
                    encoder_hidden_states = self.text_encoder(batch["text_input_ids"].to(self.accelerator.device))[0]
                
                # Predict the noise residual
                zero_timesteps = torch.zeros_like(timesteps, device=latents.device).long()
                aux = self.reference_net(
                    reference_latents,
                    zero_timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                self.to_k_hook.reverse()
                self.to_v_hook.reverse()
                
                noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states
                ).sample

                # 避免unused parameters
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean") + 0 * aux.sum()

                # loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = self.accelerator.gather(loss.repeat(self.cfg.train_batch_size)).mean().item()
                
                # Backpropagate
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                # reset hook list !!!!
                assert len(self.to_k_hook) == len(self.to_v_hook)
                assert len(self.to_k_hook) == 0
                self.to_k_hook.clear()
                self.to_v_hook.clear()

                logs = {"train/loss": avg_loss, 'train/lr': self.lr_scheduler.get_last_lr()[0]}
                self.accelerator.log(logs, step=step)
                
                if step % self.cfg.log_steps == 0:
                    if self.accelerator.is_main_process:
                        self.logger.info("Current time: {}, Step {}, lr, {}, step_loss: {}".format(
                            time.strftime("%Y-%m-%d-%H-%M-%S"), step, self.lr_scheduler.get_last_lr()[0], avg_loss))
            if step % self.cfg.save_steps == 0:
                save_path = os.path.join(self.cfg.output_dir, 'checkpoints', f"checkpoint-{step}")
                self.accelerator.save_state(save_path)

    def make_hook(self):
        # register reference_net hook
        self.to_k_hook = []
        self.to_v_hook = []

        def to_k_forward(module, input_, output_):
            self.to_k_hook.append(output_)

        def to_v_forward(module, input_, output_):
            self.to_v_hook.append(output_)
        
        for k, v in self.reference_net.named_modules():
            # attn1是self-attn, attn2是cross-attn
            # 这里注册的顺序不对，需要重新排序
            # 似乎不需要排序，forward放入list的tensor是顺序的
            if 'attn1.to_k' in k:
                # print(f'register hook for {k}, {v}')
                v.register_forward_hook(to_k_forward)
            if 'attn1.to_v' in k:
                # print(f'register hook for {k}, {v}')
                v.register_forward_hook(to_v_forward)

        def to_k2_forward(module, input_, output_):
            tmp = self.to_k_hook.pop()  # torch.float16
            assert output_.shape == tmp.shape
            # print(f'output dtype: {output_.dtype}, tmp dtype: {tmp.dtype}')
            res = torch.cat([output_, tmp], dim=1)
            return res

        def to_v2_forward(module, input_, output_):
            # output_: [b, hw, c]
            tmp = self.to_v_hook.pop()
            assert output_.shape == tmp.shape
            res = torch.cat([output_, tmp], dim=1)
            return res
        
        for k, v in self.unet.named_modules():
            if 'attn1.to_k' in k:
                # print(f'register hook for {k}, {v}')
                v.register_forward_hook(to_k2_forward)
            if 'attn1.to_v' in k:
                # print(f'register hook for {k}, {v}')
                v.register_forward_hook(to_v2_forward)
    
    def build_data(self):
        # dataloader
        train_dataset = TikTokDataset(json_file=self.cfg.json_file, tokenizer=self.tokenizer)
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train_batch_size,
            num_workers=self.cfg.dataloader_num_workers,
        )

        if self.accelerator.is_main_process:
            self.logger.info(f"Loaded {len(train_dataset)} train samples, {len(self.train_dataloader) / self.accelerator.num_processes} batches")

    def build_model(self):
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="unet")
        self.reference_net = UNet2DConditionModel.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="unet")

        # freeze parameters of models to save more memory
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        
        # TODO: 官网的教程说不需要.to(device)的操作
        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

        self.build_custom_model()
        # self.tokenizer = CLIPTokenizer.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="tokenizer")
        # self.text_encoder = CLIPTextModel.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="text_encoder")
        # self.text_encoder.requires_grad_(False)
        # self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)

    def build_custom_model(self):
        raise NotImplementedError

    def get_opt_params(self):
        return self.reference_net.parameters()

    def build_optimizer(self):
        # optimizer
        params_to_opt = self.get_opt_params()
        self.optimizer = torch.optim.AdamW(params_to_opt, lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)

        # lr scheduler
        self.lr_scheduler = get_scheduler(
            self.cfg.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.cfg.num_train_steps * self.accelerator.num_processes
        )

    def prepare(self):
        # Prepare everything with our `accelerator`.
        self.reference_net, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.reference_net, self.optimizer, self.train_dataloader, self.lr_scheduler)

    def resume(self):
        raise NotImplementedError


if __name__ == '__main__':
    pass