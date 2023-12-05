import torch
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from PIL import Image

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
    # dt_string = time.strftime("%Y-%m-%d-%H-%M")
    # # dt_string = time.strftime("%Y-%m-%d-%H-%M-%S")  # 不同进程的同步问题
    # args.output_dir = os.path.join(args.output_dir, dt_string)
    return args


def build_pipe(args, path):
    if args.control_type == 'canny':
        controlnet = ControlNetModel.from_pretrained(args.canny_controlnet)
    elif args.control_type == 'pose':
        controlnet = ControlNetModel.from_pretrained(args.pose_controlnet)
    else:
        raise NotImplementedError

    controlnet = controlnet.to(dtype=torch.float16)

    appearence_controlnet = ControlNetModel.from_pretrained(args.pose_controlnet)
    state_dict = torch.load(path, map_location='cpu')
    appearence_controlnet.load_state_dict(state_dict)
    appearence_controlnet = appearence_controlnet.to(dtype=torch.float16)
    print(f'load controlnet from {path}')

    multi_controlnet = MultiControlNetModel([appearence_controlnet, controlnet])
    
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
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        vae=infer_vae,
        controlnet=multi_controlnet,
        torch_dtype=torch.float16,
        scheduler=ddim_scheduler,
        feature_extractor=None,
        safety_checker=None
    )
    pipe.enable_model_cpu_offload()
    return pipe


def validate(args, validate_data, pipe, exp_dir, step):
    for i in range(len(validate_data.data)):
        # if i > 10:
        #     break
        # j = random.randint(0, len(validate_data.data) - 1)
        j = i
        item = validate_data.data[j]

        image = Image.open(item['image']).convert("RGB")
        reference = Image.open(item['reference']).convert("RGB")
        if args.control_type == 'canny':
            control_image = Image.open(item['canny']).convert("RGB")
        elif args.control_type == 'pose':
            control_image = Image.open(item['pose']).convert("RGB")
        else:
            raise NotImplementedError

        prompt = 'best quality'

        # TODO: support evaluate
        grid = single_image_infer(pipe, reference, prompt, control_image)
        os.makedirs(f'{exp_dir}/{step}', exist_ok=True)
        prompt = prompt.replace('/', '')[:100]
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
        grid.save(f'{exp_dir}/{step}/{prompt}_{current_time}.png')


def single_image_infer(pipe, reference, prompt, control_image):
    width, height = reference.size
    width = (width // 8) * 8
    height = (height // 8) * 8
    reference = reference.resize((width, height), Image.BILINEAR)
    control_image = control_image.resize((width, height), Image.BILINEAR)

    reference_tmp = reference
    control_image_tmp = control_image

    # reference = transforms.ToTensor()(reference)
    # reference = transforms.Normalize([0.5], [0.5])(reference)

    # control_image = transforms.ToTensor()(control_image)

    # reference = reference.unsqueeze(0)
    # control_image = control_image.unsqueeze(0)

    results = pipe(prompt=prompt, width=width, height=height, num_inference_steps=50, image=[reference, control_image], num_images_per_prompt=4).images

    all_images = [reference_tmp] + [control_image_tmp] + results
    grid = image_grid(all_images, 1, 6)
    return grid


def main():
    args = parse_args()
    pipe = build_pipe(args)
    
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    dataset = CCTVDataset(tokenizer=tokenizer)

    # metric = Metric(device='cuda')

    folder = []
    for each in os.listdir('output/exp1/2023-11-15-11-45'):
        if each.startswith('checkpoint'):
            folder.append(each)

    for each in folder:
        step = int(each.split('-')[-1])
        validate(args, dataset, pipe, ip_adapter_inference, 'output/exp1/2023-11-15-11-45', step)


if __name__ == '__main__':
    args = parse_args()
    
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    dataset = BaseDataset(json_file=args.json_file, tokenizer=tokenizer, control_type=args.control_type)
    # metric = Metric(device='cuda')

    exp_dir = args.output_dir

    # steps = list(range(0, 1000, 200))
    steps = [8000]
    for step in steps:
        pipe = build_pipe(args, os.path.join(exp_dir, f'checkpoint-{step}/pytorch_model.bin'))
        validate(args, dataset, pipe, exp_dir, step)