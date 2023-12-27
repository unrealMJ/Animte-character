import torch
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from PIL import Image

import os

from transformers import CLIPTokenizer
from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor
from con_net.model.dino import DINO

from con_net.dataset.dataset2 import LaionHumanSD, CCTVDataset, BaseDataset
from con_net.utils import copy_src, image_grid

import yaml
import argparse
import random
import time
from omegaconf import OmegaConf
import torchvision.transforms as transforms
import einops
from inference.base_inferencer import BaseInferencer
from pipeline.pipeline_stable_diffusion_image_variation_controlnet import StableDiffusionImageVariationControlNetPipeline


class Inferencer(BaseInferencer):
    def __init__(self) -> None:
        super().__init__()
        self.processor = CLIPImageProcessor.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder='feature_extractor')


    def build_pipe(self, path):
        if self.cfg.control_type == 'canny':
            controlnet = ControlNetModel.from_pretrained(self.cfg.canny_controlnet)
        elif self.cfg.control_type == 'pose':
            controlnet = ControlNetModel.from_pretrained(self.cfg.pose_controlnet)
        else:
            raise NotImplementedError

        controlnet = controlnet.to(dtype=torch.float16)

        reference_net = UNet2DConditionModel.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="unet")
        state_dict = torch.load(path, map_location='cpu')
        reference_net.load_state_dict(state_dict)
        reference_net = reference_net.to(dtype=torch.float16, device='cuda')
        self.reference_net = reference_net

        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        # vae_model_path = "/mnt/petrelfs/majie/model_checkpoint/sd-vae-ft-mse"
        # infer_vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

        pipe = StableDiffusionImageVariationControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path=self.cfg.pretrained_model_name_or_path,
            # vae=infer_vae,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            scheduler=ddim_scheduler,
            safety_checker=None
        )
        pipe.enable_model_cpu_offload()

        self.pipe = pipe

    def validate(self, exp_dir, step):
        for i in range(len(self.validate_data.data)):
            if i > 10:
                break
            item = self.validate_data.data[i]

            image = Image.open(item['image']).convert("RGB")
            reference = Image.open(item['reference']).convert("RGB")
            if self.cfg.control_type == 'canny':
                control_image = Image.open(item['canny']).convert("RGB")
            elif self.cfg.control_type == 'pose':
                control_image = Image.open(item['pose']).convert("RGB")
            else:
                raise NotImplementedError

            grid = self.single_image_infer(reference, control_image, global_image=reference)
            self.save(grid, '', exp_dir, step)

    def single_image_infer(self, reference, control_image, global_image):
        width, height = reference.size
        width = (width // 8) * 8
        height = (height // 8) * 8
        reference = reference.resize((width, height), Image.BILINEAR)
        control_image = control_image.resize((width, height), Image.BILINEAR)

        self.reference_forward(reference, global_image)

        results = self.pipe(image=global_image, control_image=control_image, width=width, height=height, num_inference_steps=50, num_images_per_prompt=4).images

        all_images = [reference] + [control_image] + results
        grid = image_grid(all_images, 1, 6)
        
        return grid


    def reference_forward(self, reference_image, global_image):
        self.reset_hook()

        reference = transforms.ToTensor()(reference_image)
        reference = transforms.Normalize([0.5], [0.5])(reference)
        reference = reference.unsqueeze(0)

        reference = reference.to('cuda').to(dtype=torch.float16)
        reference_latents = self.pipe.vae.encode(reference).latent_dist.sample()
        reference_latents = reference_latents * self.pipe.vae.config.scaling_factor

        timesteps = torch.tensor([0]).long().to('cuda')

        global_image = self.processor(global_image, return_tensors="pt").pixel_values.to('cuda', dtype=torch.float16)
        encoder_hiden_states = self.pipe.image_encoder(global_image).image_embeds  # [b, 1, 768]
        encoder_hiden_states = encoder_hiden_states.unsqueeze(1)

        _ = self.reference_net(reference_latents, timesteps, encoder_hidden_states=encoder_hiden_states)
        

if __name__ == '__main__':
    inferencer = Inferencer()
    step = 30000
    # exp_dir = inferencer.cfg.output_dir
    exp_dir = 'output/reference_clip/2023-12-20-21-32'
    # exp_dir = os.path.join(inferencer.cfg.output_dir, 'checkpoints', f'checkpoint-{step}/pytorch_model.bin')
    inferencer.build_pipe(os.path.join(exp_dir, 'checkpoints', f'checkpoint-{step}/pytorch_model.bin'))
    inferencer.make_hook()
    inferencer.validate(exp_dir, step)