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
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPVisionModel

from con_net.dataset.dataset2 import LaionHumanSD, CCTVDataset, TikTokDataset, BaseDataset, UBCFashionDataset, TikTokDataset2
from train.base_train import BaseTrainer



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
        text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
        references = torch.stack([example["reference"] for example in data])
        control_image = torch.stack([example["control_image"] for example in data])

        return {
            "images": images,
            "text_input_ids": text_input_ids,
            "references": references,
            "control_images": control_image,
        }

    def build_data(self):
        # dataloader
        train_dataset = UBCFashionDataset(json_file=self.cfg.json_file, tokenizer=self.tokenizer)
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=self.collate_fn,
            batch_size=self.cfg.train_batch_size,
            num_workers=self.cfg.dataloader_num_workers,
        )

        if self.accelerator.is_main_process:
            self.logger.info(f"Loaded {len(train_dataset)} train samples, {len(self.train_dataloader) / self.accelerator.num_processes} batches")

    def build_custom_model(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="text_encoder")
        self.text_encoder.requires_grad_(False)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)

        self.controlnet = ControlNetModel.from_pretrained(self.cfg.pose_controlnet)
        self.controlnet.requires_grad_(False)
        self.controlnet.to(self.accelerator.device, dtype=self.weight_dtype)

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
                
                # controlnet
                control_images = batch["control_images"].to(dtype=self.weight_dtype)
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=control_images,
                    return_dict=False,
                )

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
                    down_block_additional_residuals=[
                        sample.to(dtype=self.weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),
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
            if step % self.cfg.save_steps == 0:
                save_path = os.path.join(self.cfg.output_dir, 'checkpoints', f"checkpoint-{step}")
                self.accelerator.save_state(save_path)
        

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
