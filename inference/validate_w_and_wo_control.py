import torch
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from PIL import Image
from controlnet_aux import HEDdetector
from diffusers.utils import load_image
import os

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


def validate_save(grid, prompt, exp_dir, step):
    os.makedirs('results', exist_ok=True)
    prompt = prompt.replace('/', '')[:100]
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    grid.save(f'results/{prompt}_{current_time}.png')


def validate(args, pipeline, reference_net, step, to_k_hook, to_v_hook, image_path, prompt, control_path):

    reference_image = Image.open(image_path).convert("RGB")

    width = args.validate_width
    height = args.validate_height
    reference = reference_image.resize((width, height), Image.BILINEAR)
    # control_image = control_image.resize((width, height), Image.BILINEAR)
    if args.use_control:
        control_image = load_image(control_path)
        control_image = control_image.resize((width, height), Image.BILINEAR)
        if args.control_type == 'hed':
            hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
            control_image = hed(control_image)
        else:
            image = np.array(control_image)
            image = cv2.Canny(image, 100, 200)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            control_image = Image.fromarray(image)

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

    # control_image_tmp = control_image

    # reference = transforms.ToTensor()(reference)
    # reference = transforms.Normalize([0.5], [0.5])(reference)

    # control_image = transforms.ToTensor()(control_image)

    # reference = reference.unsqueeze(0)
    # control_image = control_image.unsqueeze(0)
    for i in range(len(to_k_hook)):
        feature = to_k_hook[i]
        feature = einops.repeat(feature, 'b l c -> (b n) l c', n=8)
        to_k_hook[i] = feature
    
    for i in range(len(to_v_hook)):
        feature = to_v_hook[i]
        feature = einops.repeat(feature, 'b l c -> (b n) l c', n=8)
        to_v_hook[i] = feature

    if args.use_control:
        results = pipeline(prompt=prompt, width=width, height=height, num_inference_steps=50, num_images_per_prompt=4, image=control_image).images
    else:
        results = pipeline(prompt=prompt, width=width, height=height, num_inference_steps=50, num_images_per_prompt=4).images

    # reset hook list !!!!
    assert len(to_k_hook) == len(to_v_hook)
    global k_idx, v_idx
    k_idx = 0
    v_idx = 0
    to_k_hook.clear()
    to_v_hook.clear()

    if args.use_control:
        all_images = [reference_image.resize((width, height))] + [control_image] + results
    else:
        all_images = [reference_image.resize((width, height))] + results
    grid = image_grid(all_images, 1, len(all_images))

    validate_save(grid, prompt, os.path.join(args.output_dir, 'validate'), step)


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
    # metric = Metric(device='cuda')

    exp_dir = args.output_dir

    # steps = list(range(0, 1000, 200))
    # steps = [2000, 5000, 10000]
    # for step in steps:
    #     infer = Inference(args)
    #     infer.build_pipe(os.path.join(exp_dir, f'checkpoint-{step}/pytorch_model.bin'))
    #     infer.make_hook()
    #     infer.validate(exp_dir, step)

    # single image
    step = 10000
    # infer = Inference(args)
    ckpt_path = os.path.join(exp_dir, 'pytorch_model.bin')
    # infer.build_pipe(ckpt_path)
    # infer.make_hook()
    prompt = 'a chinese boy, Song dynasty, best quality,high quality,1boy'
    prompt = 'a girl, cartoon style, pink clothes, blue hair'
    # prompt = 'best quality,high quality'
    prompt = ''


    # reference_path = '/mnt/petrelfs/majie/project/My-IP-Adapter/data/test_demo/keli/15_prompt.png'
    # control_path = '/mnt/petrelfs/majie/project/My-IP-Adapter/data/test_demo/keli/canny1.png'

    # reference_path = '/mnt/petrelfs/majie/project/My-IP-Adapter/data/test_demo/case2/reference.png'
    # control_path = '/mnt/petrelfs/majie/project/My-IP-Adapter/data/test_demo/case2/00002.png'

    reference_path = '/mnt/petrelfs/liuwenran/datasets/cctv/nantong/nantong_ref_480.jpg'
    control_path = '/mnt/petrelfs/liuwenran/datasets/cctv/nantong/nantong_ref_480.jpg'
    # reference_path = '/mnt/petrelfs/liuwenran/repos/Animte-character/data/validate_images/animation_girl.jpeg'
    # control_path = '/mnt/petrelfs/liuwenran/repos/Animte-character/data/validate_images/animation_girl.jpeg'
    # reference_path = '/mnt/petrelfs/liuwenran/repos/Animte-character/results/2023-12-25-15-36-49/2.png'
    # reference_path = '/mnt/petrelfs/liuwenran/repos/Animte-character/results/2023-12-25-15-49-46/3.png'
    control_path = '/mnt/petrelfs/liuwenran/datasets/cctv/nantong/sixpose/正面.png'
    # control_path = '/mnt/petrelfs/liuwenran/datasets/cctv/nantong/sixpose/左45.png'

    reference_net = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    state_dict = torch.load(ckpt_path, map_location='cpu')
    reference_net.load_state_dict(state_dict)
    reference_net = reference_net.to(dtype=torch.float16, device='cuda')
    
    # vae_model_path = "/mnt/petrelfs/majie/model_checkpoint/sd-vae-ft-mse"
    # infer_vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    if args.use_control:
        if args.control_type == 'canny':
            controlnet = ControlNetModel.from_pretrained(args.canny_controlnet)
        elif args.control_type == 'pose':
            controlnet = ControlNetModel.from_pretrained()
        elif args.control_type == 'hed':
            controlnet = ControlNetModel.from_pretrained(args.hed_controlnet)
        else:
            raise NotImplementedError
        
        controlnet = controlnet.to(dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
                pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                # vae=infer_vae,
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
        if ('up_blocks' in k ) and 'attn1.to_k' in k:
            print(f'register hook for {k}, {v}')
            v.register_forward_hook(to_k_forward)
        # if 'attn1.to_v' in k:
        if ('up_blocks' in k ) and 'attn1.to_v' in k:
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
        if ('up_blocks' in k ) and 'attn1.to_k' in k:
            print(f'register hook for {k}, {v}')
            v.register_forward_hook(to_k2_forward)
        # if 'attn1.to_v' in k:
        if ('up_blocks' in k ) and 'attn1.to_v' in k:
            print(f'register hook for {k}, {v}')
            v.register_forward_hook(to_v2_forward)

    validate(args, pipe, reference_net, step, to_k_hook, to_v_hook, reference_path, prompt, control_path)




