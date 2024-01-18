import os
import random
from pathlib import Path
import copy
import logging

import itertools
import time
from types import MethodType

import torch
import torch.nn.functional as F
from PIL import Image
from accelerate.logging import get_logger
from diffusers import (
    AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel, StableDiffusionControlNetPipeline,
    DDIMScheduler, StableDiffusionPipeline
    )
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPVisionModel, AutoProcessor

from con_net.dataset.dataset2 import LaionHumanSD, CCTVDataset, BaseDataset, UBCFashionDataset, TikTokDataset
from train.base_train import BaseTrainer
from con_net.model.PoseGuider import PoseGuider
from con_net.model.hack_unet2d import Hack_UNet2DConditionModel


class Trainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.init_context()
        
        self.build_model()
        self.build_optimizer()

        self.build_data()

        self.prepare()

        self.make_hook()

        self.begin_step = 0
        self.end_step = self.cfg.num_train_steps
    
    def collate_fn(self, data):
        images = torch.stack([example["image"] for example in data])
        references = torch.stack([example["reference"] for example in data])
        control_image = torch.stack([example["control_image"] for example in data])
        global_image = torch.cat([example["global_image"] for example in data], dim=0)
        return {
            "images": images,
            "global_images": global_image,
            "references": references,
            "control_images": control_image,
        }

    def build_data(self):
        # dataloader
        image_processor = AutoProcessor.from_pretrained(self.cfg.clip_vision_path)
        
        # train_dataset = UBCFashionDataset(json_file=self.cfg.json_file, tokenizer=self.tokenizer, processor=image_processor)
        train_dataset = TikTokDataset(json_file=self.cfg.json_file, tokenizer=self.tokenizer, processor=image_processor)
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train_batch_size,
            num_workers=self.cfg.dataloader_num_workers,
        )

        if self.accelerator.is_main_process:
            self.logger.info(f"Loaded {len(train_dataset)} train samples, {len(self.train_dataloader) / self.accelerator.num_processes} batches")

    def get_opt_params(self):
        return itertools.chain(
            self.reference_net.parameters(),
            self.pose_guider.parameters(),
            self.unet.parameters()
        )
    
    def build_model(self):
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="vae")
        self.unet = Hack_UNet2DConditionModel.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="unet")
        self.reference_net = UNet2DConditionModel.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="unet")

        # freeze parameters of models to save more memory
        self.vae.requires_grad_(False)
        
        # TODO: 官网的教程说不需要.to(device)的操作
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

        self.build_custom_model()

    def build_custom_model(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="tokenizer")
        
        self.image_encoder = CLIPVisionModel.from_pretrained(self.cfg.clip_vision_path)
        self.image_encoder.requires_grad_(False)
        self.image_encoder.to(self.accelerator.device, dtype=self.weight_dtype)

        self.pose_guider = PoseGuider(noise_latent_channels=320)
    
    def prepare(self):
        # Prepare everything with our `accelerator`.
        self.reference_net, self.pose_guider, self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.reference_net, self.pose_guider, self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler)
    
      
    def train(self):
        train_dataloader_iter = iter(self.train_dataloader)
        for step in range(self.begin_step, self.end_step):
            try:
                batch = next(train_dataloader_iter)
            except StopIteration:
                train_dataloader_iter = iter(self.train_dataloader)
                batch = next(train_dataloader_iter)
            
            with self.accelerator.accumulate(self.reference_net, self.pose_guider, self.unet):
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
                    encoder_hidden_states = self.image_encoder(batch["global_images"].to(self.accelerator.device)).pooler_output
                    encoder_hidden_states = encoder_hidden_states.unsqueeze(1)  # [bs, 1, 768]
                
                # pose guider
                control_images = batch["control_images"].to(dtype=self.weight_dtype)
                pose_latents = self.pose_guider(control_images)

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
                    encoder_hidden_states=encoder_hidden_states,
                    latent_pose=pose_latents
                ).sample

                # 避免unused parameters
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean") + 0 * aux.sum()

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
            if step % self.cfg.save_steps == 0 or (step < 2500 and step % 500 == 0):
                save_path = os.path.join(self.cfg.output_dir, 'checkpoints', f"checkpoint-{step}")
                self.accelerator.save_state(save_path)


    def make_hook2(self):
        # register reference_net hook
        self.to_k_hook = []
        self.to_v_hook = []

        def to_k_forward(module, input_):
            self.to_k_hook.append(input_)

        def to_v_forward(module, input_):
            self.to_v_hook.append(input_)
        
        for k, v in self.reference_net.named_modules():
            # attn1是self-attn, attn2是cross-attn
            # 这里注册的顺序不对，需要重新排序
            # 似乎不需要排序，forward放入list的tensor是顺序的
            if 'attn1.to_k' in k:
                v.register_forward_pre_hook(to_k_forward)
            if 'attn1.to_v' in k:
                v.register_forward_pre_hook(to_v_forward)

        def to_k2_forward(module, input_):
            tmp = self.to_k_hook.pop()  # torch.float16
            assert input_.shape == tmp.shape
            res = torch.cat([input_, tmp], dim=1)
            return res

        def to_v2_forward(module, input_):
            # output_: [b, hw, c]
            tmp = self.to_v_hook.pop()
            assert input_.shape == tmp.shape
            res = torch.cat([input_, tmp], dim=1)
            return res
        
        for k, v in self.unet.named_modules():
            if 'attn1.to_k' in k:
                v.register_forward_pre_hook(to_k_forward)
            if 'attn1.to_v' in k:
                v.register_forward_pre_hook(to_v_forward)
    
    def make_hook(self):
        # register reference_net hook
        self.to_k_hook = []
        self.to_v_hook = []

        def to_k_forward(module, input_, output_):
            self.to_k_hook.append(output_)

        def to_v_forward(module, input_, output_):
            self.to_v_hook.append(output_)
        
        for k, v in self.reference_net.named_modules():
            if 'attn1.to_k' in k:
                v.register_forward_hook(to_k_forward)
            if 'attn1.to_v' in k:
                v.register_forward_hook(to_v_forward)

        def to_k2_forward(module, input_, output_):
            tmp = self.to_k_hook.pop()  # torch.float16
            assert output_.shape == tmp.shape
            res = torch.cat([output_, tmp], dim=1)
            return res

        def to_v2_forward(module, input_, output_):
            tmp = self.to_v_hook.pop()
            assert output_.shape == tmp.shape
            res = torch.cat([output_, tmp], dim=1)
            return res
        
        for k, v in self.unet.named_modules():
            if 'attn1.to_k' in k:
                v.register_forward_hook(to_k2_forward)
            if 'attn1.to_v' in k:
                v.register_forward_hook(to_v2_forward)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
