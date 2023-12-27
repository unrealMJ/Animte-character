import torch
from diffusers import (
    StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel, 
    StableDiffusionPipeline, UNet2DConditionModel, MotionAdapter)
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.utils import export_to_gif
from PIL import Image

import os

from transformers import CLIPTokenizer

from con_net.dataset.dataset2 import LaionHumanSD, CCTVDataset, BaseDataset
from con_net.utils import copy_src, image_grid
from pipeline.pipeline_animatediff_controlnet import AnimateDiffControlNetPipeline

import argparse
import random
import time
from omegaconf import OmegaConf
import torchvision.transforms as transforms
import einops
import numpy as np


class Inference:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        
        self.tokenizer = CLIPTokenizer.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="tokenizer")
        self.validate_data = self.construct_data()
        
        self.to_k_hook = []
        self.to_v_hook = []

        self.k_idx = 0
        self.v_idx = 0

    def build_pipe(self, reference_path, motion_path):
        if self.cfg.control_type == 'canny':
            controlnet = ControlNetModel.from_pretrained(self.cfg.canny_controlnet)
        elif self.cfg.control_type == 'pose':
            controlnet = ControlNetModel.from_pretrained(self.cfg.pose_controlnet)
        else:
            raise NotImplementedError

        controlnet = controlnet.to(dtype=torch.float16)

        reference_net = UNet2DConditionModel.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="unet")
        state_dict = torch.load(reference_path, map_location='cpu')
        reference_net.load_state_dict(state_dict)
        reference_net = reference_net.to(dtype=torch.float16, device='cuda')
        
        # motion_adapter = MotionAdapter.from_pretrained(args.motion_path)
        motion_adapter = MotionAdapter.from_config(self.cfg.motion_path)
        motion_state = torch.load(motion_path, map_location='cpu')
        # TODO: check
        motion_adapter.load_state_dict(motion_state, strict=False)
        motion_adapter = motion_adapter.to(dtype=torch.float16, device='cuda')


        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        pipe = AnimateDiffControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path=self.cfg.pretrained_model_name_or_path,
            motion_adapter=motion_adapter,
            # vae=infer_vae,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            scheduler=ddim_scheduler,
            feature_extractor=None,
            safety_checker=None
        )

        pipe.enable_model_cpu_offload()
        self.pipe = pipe
        self.reference_net = reference_net

    def construct_data(self):
        tokenizer = CLIPTokenizer.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="tokenizer")
        dataset = BaseDataset(json_file=self.cfg.json_file, tokenizer=tokenizer, control_type=self.cfg.control_type)
        return dataset

    def validate(self, exp_dir, step):
        sample_n_frames = 24
        sample_stride = 4

        for i in range(len(self.validate_data.data)):
            item = self.validate_data.data[i]

            role_root = item['role_root']
            all_images = os.listdir(f'{role_root}/images')
            video_length = len(all_images)

            clip_length = min(video_length, (sample_n_frames - 1) * sample_stride + 1)
            start_idx = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, sample_n_frames, dtype=int)

            control_list = []
            for idx in batch_index:
                idx = str(idx + 1).zfill(4)
                control_image = Image.open(f'{role_root}/{self.control_type}/{idx}.png').convert("RGB")
                control_list.append(control_image)

            idx = batch_index[0] + 1
            reference = Image.open(f'{role_root}/images/{idx}.png').convert("RGB")

            prompt = 'best quality,high quality'
            grid = self.single_image_infer(reference, prompt, control_image)
            self.save(grid, prompt, exp_dir, step)
    
    def save(self, reference, control_video, result, prompt, exp_dir, step):
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
        save_dir = f'{exp_dir}/{step}/{current_time}'
        os.makedirs(save_dir, exist_ok=True)
        prompt = prompt.replace('/', '')[:100]
        reference.save(f'{save_dir}/reference.png')
        export_to_gif(control_video, f'{save_dir}/control.gif')
        export_to_gif(result, f'{save_dir}/{prompt}.gif')

        os.makedirs(f'{save_dir}/frames', exist_ok=True)
        for i in range(len(result)):
            result[i].save(f'{save_dir}/frames/{i}.png')


    def single_video_infer(self, reference, prompt, control_video):
        width, height = reference.size
        width = (width // 8) * 8
        height = (height // 8) * 8
        reference = reference.resize((width, height), Image.BILINEAR)

        control_video = [each.resize((width, height), Image.BILINEAR) for each in control_video]

        self.reference_forward(reference, prompt)

        reference_tmp = reference
        # control_image_tmp = control_image

        # reference = transforms.ToTensor()(reference)
        # reference = transforms.Normalize([0.5], [0.5])(reference)

        # control_image = transforms.ToTensor()(control_image)

        # reference = reference.unsqueeze(0)
        # control_image = control_image.unsqueeze(0)

        results = self.pipe(prompt=prompt, width=width, height=height, num_frames=self.cfg.num_frames, num_inference_steps=50, image=control_video).frames[0]
        return results
    
    def reference_forward(self, reference_image, prompt):
        self.reset_hook()
        reference = transforms.ToTensor()(reference_image)
        reference = transforms.Normalize([0.5], [0.5])(reference)
        reference = reference.unsqueeze(0)

        reference = reference.to('cuda').to(dtype=torch.float16)
        reference_latents = self.pipe.vae.encode(reference).latent_dist.sample()
        reference_latents = reference_latents * self.pipe.vae.config.scaling_factor

        text_input_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        text_input_ids = text_input_ids[None].to('cuda')

        encoder_hidden_states = self.pipe.text_encoder(text_input_ids)[0]
        
        timesteps = torch.tensor([0]).long().to('cuda')

        _ = self.reference_net(reference_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

    def make_hook(self):
        def to_k_forward(module, input_, output_):
            # 这里的n是batch size，由于有cfg所以乘2
            tmp = einops.repeat(output_, 'b l c -> (b n) l c', n=2 * self.cfg.num_frames)
            self.to_k_hook.append(tmp)

        def to_v_forward(module, input_, output_):
            tmp = einops.repeat(output_, 'b l c -> (b n) l c', n=2 * self.cfg.num_frames)
            self.to_v_hook.append(tmp)
        
        for k, v in self.reference_net.named_modules():
            # attn1是self-attn, attn2是cross-attn
            # 这里注册的顺序不对，需要重新排序
            # 似乎不需要排序，forward放入list的tensor是顺序的
            if 'attn1.to_k' in k:
                print(f'register hook for {k}, {v}')
                v.register_forward_hook(to_k_forward)
            if 'attn1.to_v' in k:
                print(f'register hook for {k}, {v}')
                v.register_forward_hook(to_v_forward)
        
        #  ############
        
        def to_k2_forward(module, input_, output_):
            # output_: [b, hw, c]
            try:
                tmp = self.to_k_hook[self.k_idx]  # [b*f, h*w, c]
            except:
                self.k_idx = 0
                tmp = self.to_k_hook[self.k_idx]
            self.k_idx += 1
            assert output_.shape == tmp.shape, f'{output_.shape}, {tmp.shape}'
            res = torch.cat([output_, tmp], dim=1)
            return res

        def to_v2_forward(module, input_, output_):
            try:
                tmp = self.to_v_hook[self.v_idx]
            except:
                self.v_idx = 0
                tmp = self.to_v_hook[self.v_idx]
            self.v_idx += 1
            assert output_.shape == tmp.shape, f'{output_.shape}, {tmp.shape}'
            res = torch.cat([output_, tmp], dim=1)
            return res
        
        for k, v in self.pipe.unet.named_modules():
            if 'attn1.to_k' in k and 'motion_modules' not in k:
                print(f'register hook for {k}, {v}')
                v.register_forward_hook(to_k2_forward)
            if 'attn1.to_v' in k and 'motion_modules' not in k:
                print(f'register hook for {k}, {v}')
                v.register_forward_hook(to_v2_forward)
        
    def reset_hook(self):
        assert self.k_idx == len(self.to_k_hook)
        assert self.v_idx == len(self.to_v_hook)
        self.k_idx = 0
        self.v_idx = 0
        self.to_k_hook.clear()
        self.to_v_hook.clear()



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
    exp_dir = args.output_dir

    # single image
    step = 25000
    infer = Inference(args)
    reference_path = '/mnt/petrelfs/majie/project/My-IP-Adapter/output/reference_net/2023-12-04-17-27/checkpoint-30000/pytorch_model.bin'
    motion_path = os.path.join(exp_dir, f'checkpoint-{step}/pytorch_model.bin')

    infer.build_pipe(reference_path, motion_path)
    infer.make_hook()
    prompt = 'best quality,high quality'

    # load data
    sample_n_frames = args.num_frames
    sample_stride = 4
    # role_root = '/mnt/petrelfs/majie/project/diffusers/my_project/identity/TikTok/TikTok_dataset/00122'
    role_root = '/mnt/petrelfs/majie/project/diffusers/my_project/identity/aux_data/wechat-group/1'
    all_images = os.listdir(f'{role_root}/images')
    video_length = len(all_images)
    
    clip_length = min(video_length, (sample_n_frames - 1) * sample_stride + 1)
    start_idx = random.randint(0, video_length - clip_length)
    # start_idx = 40
    batch_index = np.linspace(start_idx, start_idx + clip_length - 1, sample_n_frames, dtype=int)

    control_video = []
    for idx in batch_index:
        idx = str(idx + 1).zfill(4)
        control_image = Image.open(f'{role_root}/pose/{idx}.png').convert("RGB")
        control_video.append(control_image)

    # TODO: 检查 +1的正确性
    idx = str(batch_index[0] + 1).zfill(4)
    reference = Image.open(f'{role_root}/images/{idx}.png').convert("RGB")

    result = infer.single_video_infer(reference, prompt, control_video)

    infer.save(reference, control_video, result, prompt, exp_dir, step)
    # infer.save(grid, prompt, exp_dir, step)
