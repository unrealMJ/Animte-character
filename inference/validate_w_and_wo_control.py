import torch
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from PIL import Image
from controlnet_aux import HEDdetector
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image
import os
from safetensors.torch import load_file
from transformers import CLIPTokenizer

from con_net.dataset.dataset2 import LaionHumanSD, CCTVDataset, BaseDataset
from con_net.utils import copy_src, image_grid

import yaml
import argparse
import random
import time
from omegaconf import OmegaConf
import torchvision.transforms as transforms
import einops
import numpy as np
import cv2


def validate_save(grid, exp_dir, validation_id):
    exp_dir_split = exp_dir.split('/')
    exp_name = exp_dir_split[-3] + '_' + exp_dir_split[-2] + '_' + exp_dir_split[-1]
    save_dir = os.path.join('results', exp_name, validation_id)
    os.makedirs(save_dir, exist_ok=True)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    grid.save(f'{save_dir}/{current_time}.png')


def validate(args, pipeline, reference_net, to_k_hook, to_v_hook, use_control, control_type, validation_id):
    val_image_lines = open(args.validate_image_file).readlines()
    val_control_image_lines = open(args.validate_control_image_file).readlines()

    if use_control:
        if control_type == 'hed':
            hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
        elif control_type == 'pose':
            processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

    for i in range(len(val_image_lines)):
        image = val_image_lines[i].strip()
        # prompt = val_prompt_lines[i].strip()
        prompt = ''

        reference_image = Image.open(image).convert("RGB")

        width = args.validate_width
        height = args.validate_height
        reference = reference_image.resize((width, height), Image.BILINEAR)
        if use_control:
            control_image_line = val_control_image_lines[i].strip()
            control_image = Image.open(control_image_line).convert("RGB")
            control_image = control_image.resize((width, height), Image.BILINEAR)

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
        reference_net(reference_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

        for i in range(len(to_k_hook)):
            feature = to_k_hook[i]
            feature = einops.repeat(feature, 'b l c -> (b n) l c', n=8)
            to_k_hook[i] = feature
        
        for i in range(len(to_v_hook)):
            feature = to_v_hook[i]
            feature = einops.repeat(feature, 'b l c -> (b n) l c', n=8)
            to_v_hook[i] = feature


        if use_control:
            if control_type == 'canny':
                control_image = np.array(control_image)
                control_image = cv2.Canny(control_image, 100, 200)
                control_image = control_image[:, :, None]
                control_image = np.concatenate([control_image, control_image, control_image], axis=2)
                control_image = Image.fromarray(control_image)
            elif control_type == 'pose':
                control_image = processor(control_image, hand_and_face=True)
            elif control_type == 'hed':
                control_image = hed(control_image)
            with torch.autocast("cuda"):
                results = pipeline(prompt=prompt, width=width, height=height, image=control_image, num_inference_steps=50, num_images_per_prompt=4).images
        else:
            with torch.autocast("cuda"):
                results = pipeline(prompt=prompt, width=width, height=height, num_inference_steps=50, num_images_per_prompt=4).images

        # reset hook list !!!!
        assert len(to_k_hook) == len(to_v_hook)
        global k_idx, v_idx
        k_idx = 0
        v_idx = 0
        to_k_hook.clear()
        to_v_hook.clear()

        if use_control:
            all_images = [reference_image.resize((width, height))] + [control_image] + results
        else:
            all_images = [reference_image.resize((width, height))] + results
        grid = image_grid(all_images, 1, len(all_images))

        validate_save(grid, args.exp_dir, validation_id)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default='config/inference/base.yaml',
    )
    
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    # TODO: check argparse.Namespace 的使用方法
    # args = argparse.Namespace(**cfg, **vars(args))
    for k, v in cfg.items():
        setattr(args, k, v)
    return args


if __name__ == '__main__':
    args = parse_args()

    exp_dir = args.exp_dir

    reference_net = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    # state_dict = torch.load(ckpt_path, map_location='cpu')
    # reference_net.load_state_dict(state_dict)
    reference_net_ckpt_path = os.path.join(exp_dir, 'model.safetensors')
    reference_net_state_dict = load_file(reference_net_ckpt_path)
    reference_net.load_state_dict(reference_net_state_dict)
    reference_net = reference_net.to(dtype=torch.float16, device='cuda')
    
    infer_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch.float16)
    if args.use_control:
        if args.control_type == 'canny':
            controlnet = ControlNetModel.from_pretrained(args.canny_controlnet)
        elif args.control_type == 'pose':
            controlnet = ControlNetModel.from_pretrained(args.pose_controlnet)
        elif args.control_type == 'hed':
            controlnet = ControlNetModel.from_pretrained(args.hed_controlnet)
        else:
            raise NotImplementedError
        
        controlnet_ckpt_path = os.path.join(exp_dir, 'model_1.safetensors')
        controlnet_state_dict = load_file(controlnet_ckpt_path)
        controlnet.load_state_dict(controlnet_state_dict)
        controlnet = controlnet.to(dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
                pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                vae=infer_vae,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                feature_extractor=None,
                safety_checker=None
            ).to(dtype=torch.float16, device='cuda')
        pipe.enable_model_cpu_offload()
        
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            torch_dtype=torch.float16,
            feature_extractor=None,
            safety_checker=None
        ).to(dtype=torch.float16, device='cuda')
        pipe.enable_model_cpu_offload()

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
        # if 'attn1.to_k' in k:
        if ('up_blocks' in k or 'mid_block' in k) and 'attn1.to_k' in k:
            print(f'register hook for {k}, {v}')
            v.register_forward_hook(to_k_forward)
        # if 'attn1.to_v' in k:
        if ('up_blocks' in k or 'mid_block' in k) and 'attn1.to_v' in k:
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

    for k, v in pipe.unet.named_modules():
        # attn1是self-attn, attn2是cross-attn
        # 这里注册的顺序不对，需要重新排序
        # if 'attn1.to_k' in k:
        if ('up_blocks' in k or 'mid_block' in k) and 'attn1.to_k' in k:
            print(f'register hook for {k}, {v}')
            v.register_forward_hook(to_k2_forward)
        # if 'attn1.to_v' in k:
        if ('up_blocks' in k or 'mid_block' in k) and 'attn1.to_v' in k:
            print(f'register hook for {k}, {v}')
            v.register_forward_hook(to_v2_forward)

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    validation_id = current_time
    validate(args, pipe, reference_net, to_k_hook, to_v_hook, args.use_control, args.control_type, validation_id)
