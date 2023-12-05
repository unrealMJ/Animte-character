from types import MethodType

import torch
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel, StableDiffusionPipeline
from PIL import Image

from ip_adapter import IPAdapter
from ip_adapter.utils import generate
import os

from transformers import CLIPTokenizer

from con_net.dataset.dataset2 import LaionHumanSD
from con_net.metric import Metric
from con_net.utils import copy_src, image_grid

import yaml
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default='config/base.yaml',
    )
    parser.add_argument(
        "--ip_ckpt",
        type=str,
        default='IP-Adapter/models/ip-adapter_sd15.bin',
        help="Path to CLIP image encoder",
    )
    
    args = parser.parse_args()

    # merge yaml cfg to args
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    # TODO: check argparse.Namespace 的使用方法
    # args = argparse.Namespace(**cfg, **vars(args))
    for k, v in cfg.items():
        setattr(args, k, v)

    # TODO: parse yaml with OmegaConf
    args.learning_rate = float(args.learning_rate)
    args.num_train_steps = int(float(args.num_train_steps))

    # dt_string = time.strftime("%Y-%m-%d-%H-%M")
    # # dt_string = time.strftime("%Y-%m-%d-%H-%M-%S")  # 不同进程的同步问题
    # args.output_dir = os.path.join(args.output_dir, dt_string)
    return args


def build_pipe(args):
    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    sd_pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float32,
        scheduler=ddim_scheduler,
        feature_extractor=None,
        safety_checker=None
    )

    vae_model_path = "/mnt/petrelfs/majie/model_checkpoint/sd-vae-ft-mse"
    infer_vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    ip_pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        scheduler=ddim_scheduler,
        vae=infer_vae,
        feature_extractor=None,
        safety_checker=None
    )
    ip_pipe.__call__ = MethodType(generate, ip_pipe)

    ip_adapter_inference = IPAdapter(sd_pipe=ip_pipe, image_encoder_path=args.image_encoder_path,
                            ip_ckpt=None, device='cuda')
    
    return sd_pipe, ip_adapter_inference


def load_ip_ckpt(ip_adapter_inference, ip_ckpt):
    state_dict = torch.load(ip_ckpt, map_location='cpu')
    image_proj_state_dict = {}
    ip_state_dict = {}
    for k, v in state_dict.items():
        if 'image_proj_model.' in k:
            image_proj_state_dict[k.replace('image_proj_model.', '')] = v
        elif 'adapter_modules.' in k:
            ip_state_dict[k.replace('adapter_modules.', '')] = v
        else:
            pass
    
    ip_adapter_inference.image_proj_model.load_state_dict(image_proj_state_dict)
    ip_layers = torch.nn.ModuleList(ip_adapter_inference.pipe.unet.attn_processors.values())
    ip_layers.load_state_dict(ip_state_dict)
    return ip_adapter_inference

def validate(args, validate_data, sd_pipe, ip_adapter_inference, exp_dir, step):
    ip_ckpt = f'{exp_dir}/checkpoint-{step}/pytorch_model.bin'
    ip_adapter_inference = load_ip_ckpt(ip_adapter_inference, ip_ckpt)

    for i in range(len(validate_data.data)):
        if i > 10:
            break
        j = random.randint(0, len(validate_data.data) - 1)
        each = validate_data.data[j]

        key = each['key']
        try:
            clip_image = Image.open(each['img_path']).convert("RGB")
        except Exception:
            continue
        prompt = each['prompt']

        # TODO: support evaluate
        grid = single_image_infer(sd_pipe, ip_adapter_inference, clip_image, prompt)
        os.makedirs(f'{exp_dir}/{step}', exist_ok=True)
        prompt = prompt.replace('/', '')[:100]
        grid.save(f'{exp_dir}/{step}/{prompt}.png')



def single_image_infer(sd_pipe, ip_adapter_inference, clip_image, prompt):
    width, height = clip_image.size
    width = (width // 8) * 8
    height = (height // 8) * 8
    clip_image = clip_image.resize((width, height), Image.BILINEAR)

    # sd_result = sd_pipe(prompt=prompt, width=width, height=height, num_inference_steps=50).images
    ip_result = ip_adapter_inference.generate(pil_image=clip_image, prompt=prompt, num_samples=2, width=width,
                                 height=height, num_inference_steps=50)
    gen_propmt = 'best quality, high quality'
    ip_no_prompt_result = ip_adapter_inference.generate(pil_image=clip_image, prompt=gen_propmt, num_samples=2, width=width,
                                             height=height, num_inference_steps=50) 

    all_images = [clip_image] + ip_no_prompt_result + ip_result
    grid = image_grid(all_images, 1, 5)
    return grid

def main():
    args = parse_args()
    pipe, ip_adapter_inference = build_pipe(args)
    
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    dataset = LaionHumanSD(json_file=args.json_file, tokenizer=tokenizer)

    # metric = Metric(device='cuda')

    folder = []
    for each in os.listdir('output/exp1/2023-11-15-11-45'):
        if each.startswith('checkpoint'):
            folder.append(each)

    for each in folder:
        step = int(each.split('-')[-1])
        validate(args, dataset, pipe, ip_adapter_inference, 'output/exp1/2023-11-15-11-45', step)


def main2():
    args = parse_args()
    pipe, ip_adapter_inference = build_pipe(args)

    exp_dir = 'output/exp1/2023-11-15-11-45'
    step = 10000

    ip_ckpt = f'{exp_dir}/checkpoint-{step}/pytorch_model.bin'
    ip_adapter_inference = load_ip_ckpt(ip_adapter_inference, ip_ckpt)


    pass


if __name__ == '__main__':
    args = parse_args()
    pipe, ip_adapter_inference = build_pipe(args)
    
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    dataset = LaionHumanSD(json_file=args.json_file, tokenizer=tokenizer)

    # metric = Metric(device='cuda')

    steps = []
    for each in os.listdir('output/exp1/2023-11-16-10-15'):
        if each.startswith('checkpoint'):
            steps.append(int(each.split('-')[-1]))

    steps = [30000, 40000, 44000, 48000]
    for step in steps:
        validate(args, dataset, pipe, ip_adapter_inference, 'output/exp1/2023-11-16-10-15', step)