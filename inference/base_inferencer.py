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
import einops


class BaseInferencer:
    def __init__(self) -> None:
        self.cfg = self.parse_args()
        
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="tokenizer")
        except:
            self.tokenizer = None
        self.validate_data = self.construct_data()
        
        self.to_k_hook = []
        self.to_v_hook = []

        self.k_idx = 0
        self.v_idx = 0

    def build_pipe(self):
        if self.cfg.control_type == 'canny':
            controlnet = ControlNetModel.from_pretrained(self.cfg.canny_controlnet)
        elif self.cfg.control_type == 'pose':
            controlnet = ControlNetModel.from_pretrained(self.cfg.pose_controlnet)
        else:
            raise NotImplementedError

        controlnet = controlnet.to(dtype=torch.float16)

        reference_net = UNet2DConditionModel.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="unet")
        path = os.path.join(self.cfg.output_dir, f'checkpoints/checkpoint-{self.cfg.step}/pytorch_model.bin')
        state_dict = torch.load(path, map_location='cpu')
        reference_net.load_state_dict(state_dict)
        reference_net = reference_net.to(dtype=torch.float16, device='cuda')
        
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

        self.pipe = pipe
        self.reference_net = reference_net

    def construct_data(self):
        dataset = BaseDataset(json_file=self.cfg.json_file, tokenizer=self.tokenizer, control_type=self.cfg.control_type)
        return dataset

    def validate(self):
        exp_dir = self.cfg.output_dir
        step = self.cfg.step
        for i in range(len(self.validate_data.data)):
            item = self.validate_data.data[i]

            image = Image.open(item['image']).convert("RGB")
            reference = Image.open(item['reference']).convert("RGB")
            if self.cfg.control_type == 'canny':
                control_image = Image.open(item['canny']).convert("RGB")
            elif self.cfg.control_type == 'pose':
                control_image = Image.open(item['pose']).convert("RGB")
            else:
                raise NotImplementedError

            prompt = 'best quality'
            grid = self.single_image_infer(reference, prompt, control_image)
            self.save(grid, prompt, exp_dir, step)
    
    def save(self, grid, filename, exp_dir, step):
        os.makedirs(f'{exp_dir}/{step}', exist_ok=True)
        filename = filename.replace('/', '')[:100]
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
        grid.save(f'{exp_dir}/{step}/{current_time}_{filename}.png')

    def single_image_infer(self, reference, prompt, control_image, return_raw=False):
        width, height = reference.size
        width = (width // 8) * 8
        height = (height // 8) * 8
        reference = reference.resize((width, height), Image.BILINEAR)
        control_image = control_image.resize((width, height), Image.BILINEAR)

        self.reference_forward(reference, prompt)

        results = self.pipe(prompt=prompt, width=width, height=height, num_inference_steps=50, image=control_image, num_images_per_prompt=4).images

        all_images = [reference] + [control_image] + results
        grid = image_grid(all_images, 1, 6)
        
        if return_raw:
            return all_images
        else:
            return grid
    
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
            # 这里的n是batch size，输出4张图，由于有cfg所以是8
            tmp = einops.repeat(output_, 'b l c -> (b n) l c', n=8)
            self.to_k_hook.append(tmp)

        def to_v_forward(module, input_, output_):
            tmp = einops.repeat(output_, 'b l c -> (b n) l c', n=8)
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
            try:
                tmp = self.to_k_hook[self.k_idx]
            except:
                self.k_idx = 0
                tmp = self.to_k_hook[self.k_idx]
            self.k_idx += 1
            assert output_.shape == tmp.shape, f'{output_.shape}, {tmp.shape}'
            res = torch.cat([output_, tmp], dim=1)
            return res

        def to_v2_forward(module, input_, output_):
            # output_: [b, hw, c]
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
            if 'attn1.to_k' in k:
                print(f'register hook for {k}, {v}')
                v.register_forward_hook(to_k2_forward)
            if 'attn1.to_v' in k:
                print(f'register hook for {k}, {v}')
                v.register_forward_hook(to_v2_forward)
        
    def reset_hook(self):
        assert self.k_idx == len(self.to_k_hook)
        assert self.v_idx == len(self.to_v_hook)
        self.k_idx = 0
        self.v_idx = 0
        # assert len(self.to_k_hook) == 0
        # assert len(self.to_v_hook) == 0
        self.to_k_hook.clear()
        self.to_v_hook.clear()

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Simple example of a training script.")
        parser.add_argument(
            "--config",
            type=str,
            default='config/inference/base.yaml',
        )
        
        # args = parser.parse_args()
        args, unknown = parser.parse_known_args()

        cfg = OmegaConf.load(args.config)
        if unknown:
            for arg in unknown:
                if '=' not in arg:
                    continue
                key, value = arg.split('=')
                # print(f'update {key} from {getattr(cfg, key)} to {value}')
                setattr(cfg, key, value)

        if hasattr(cfg, 'step'):
            assert type(cfg.step) == str, f'{type(cfg.step)}'
            cfg.step = int(cfg.step)

        # TODO: check argparse.Namespace 的使用方法
        # args = argparse.Namespace(**cfg, **vars(args))
        for k, v in cfg.items():
            setattr(args, k, v)
        return args


if __name__ == '__main__':
    args = parse_args()
    # metric = Metric(device='cuda')

    exp_dir = args.output_dir

    steps = list(range(0, 1000, 200))
    steps = [25000, 20000]
    for step in steps:
        infer = Inference(args)
        infer.build_pipe(os.path.join(exp_dir, f'checkpoints/checkpoint-{step}/pytorch_model.bin'))
        infer.make_hook()
        infer.validate(exp_dir, step)

    # single image
    # step = 21000
    # infer = Inference(args)
    # infer.build_pipe(os.path.join(exp_dir, f'checkpoint-{step}/pytorch_model.bin'))
    # infer.make_hook()
    # prompt = '1girl,upper body,cry,sad'

    # reference_path = '/mnt/petrelfs/majie/project/My-IP-Adapter/data/test_demo/0.png'
    # control_path = '/mnt/petrelfs/majie/project/My-IP-Adapter/data/test_demo/pose2.png'

    # reference = Image.open(reference_path).convert("RGB")
    # control_image = Image.open(control_path).convert("RGB")

    # grid = infer.single_image_infer(reference, prompt, control_image)
    # infer.save(grid, prompt, exp_dir, step)
