import torch
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from PIL import Image

import os

from transformers import CLIPTokenizer
from transformers import AutoImageProcessor, AutoModel
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


class Inferencer(BaseInferencer):
    def __init__(self) -> None:
        super().__init__()
        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-giant')

    def build_pipe(self, path):
        if self.cfg.control_type == 'canny':
            controlnet = ControlNetModel.from_pretrained(self.cfg.canny_controlnet)
        elif self.cfg.control_type == 'pose':
            controlnet = ControlNetModel.from_pretrained(self.cfg.pose_controlnet)
        else:
            raise NotImplementedError

        controlnet = controlnet.to(dtype=torch.float16)

        reference_weight_path = os.path.join(path, 'pytorch_model.bin')
        reference_net = UNet2DConditionModel.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="unet")
        state_dict = torch.load(reference_weight_path, map_location='cpu')
        reference_net.load_state_dict(state_dict)
        reference_net = reference_net.to(dtype=torch.float16, device='cuda')
        self.reference_net = reference_net

        image_encoder = DINO()
        linear_weight_path = os.path.join(path, 'pytorch_model_1.bin')
        state_dict = torch.load(linear_weight_path, map_location='cpu')
        image_encoder.linear.load_state_dict(state_dict)
        image_encoder = image_encoder.to(dtype=torch.float16, device='cuda')
        self.image_encoder = image_encoder

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

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path=self.cfg.pretrained_model_name_or_path,
            # vae=infer_vae,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            scheduler=ddim_scheduler,
            feature_extractor=None,
            safety_checker=None
        )
        pipe.enable_model_cpu_offload()

        pipe.text_encoder = None
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

            grid = self.single_image_infer(reference, control_image, dino_image=reference)
            self.save(grid, '', exp_dir, step)

    def single_image_infer(self, reference, control_image, dino_image):
        width, height = reference.size
        width = (width // 8) * 8
        height = (height // 8) * 8
        reference = reference.resize((width, height), Image.BILINEAR)
        control_image = control_image.resize((width, height), Image.BILINEAR)

        dino_image = self.dino_processor(dino_image, return_tensors="pt").pixel_values.to('cuda', dtype=torch.float16)
        prompt_embeds = self.image_encoder(dino_image)  # [b, h*w, 768]
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)

        self.reference_forward(reference, prompt_embeds, negative_prompt_embeds)

        results = self.pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, width=width, height=height, num_inference_steps=50, image=control_image, num_images_per_prompt=4).images

        all_images = [reference] + [control_image] + results
        grid = image_grid(all_images, 1, 6)
        
        return grid


    def reference_forward(self, reference_image, prompt_embeds, negative_prompt_embeds):
        self.reset_hook()

        reference = transforms.ToTensor()(reference_image)
        reference = transforms.Normalize([0.5], [0.5])(reference)
        reference = reference.unsqueeze(0)

        reference = reference.to('cuda').to(dtype=torch.float16)
        reference_latents = self.pipe.vae.encode(reference).latent_dist.sample()
        reference_latents = reference_latents * self.pipe.vae.config.scaling_factor

        timesteps = torch.tensor([0]).long().to('cuda')

        _ = self.reference_net(reference_latents, timesteps, encoder_hidden_states=prompt_embeds)

    def save(self, grid, filename, exp_dir, step):
        os.makedirs(f'{exp_dir}/{step}', exist_ok=True)
        filename = filename.replace('/', '')[:100]
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
        grid.save(f'{exp_dir}/{step}/{current_time}_{filename}.png')


if __name__ == '__main__':
    inferencer = Inferencer()
    step = 25000
    exp_dir = inferencer.cfg.output_dir
    # exp_dir = os.path.join(inferencer.cfg.output_dir, 'checkpoints', f'checkpoint-{step}/pytorch_model.bin')
    inferencer.build_pipe(os.path.join(exp_dir, 'checkpoints', f'checkpoint-{step}'))
    inferencer.make_hook()
    inferencer.validate(exp_dir, step)