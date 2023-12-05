import os
import random
import argparse
from pathlib import Path
import copy
import logging

import itertools
import time
from types import MethodType

import torch
import torch.nn.functional as F
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import (
    AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel, StableDiffusionControlNetPipeline,
    DDIMScheduler, StableDiffusionPipeline
    )
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPVisionModel

from omegaconf import OmegaConf

device = 'cuda'

unet1 = UNet2DConditionModel.from_pretrained('/mnt/petrelfs/majie/model_checkpoint/stable-diffusion-v1-5', subfolder="unet").to(device)
unet2 = UNet2DConditionModel.from_pretrained('/mnt/petrelfs/majie/model_checkpoint/stable-diffusion-v1-5', subfolder="unet").to(device)
noisy_latents = torch.randn(2, 4, 64, 64).to(device)
timesteps = torch.randint(0, 1000, (2,), device=device)
encoder_hidden_states = torch.randn(2, 77, 768).to(device)

to_k_hook = []
to_v_hook = []

# register hook
def to_k_forward(module, input_, output_):
    to_k_hook.append(output_) # 然后分别存入全局 list 中

def to_v_forward(module, input_, output_):
    to_v_hook.append(output_) # 然后分别存入全局 list 中

# unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1.to_k.register_forward_hook(lambda self, input, output: to_k_hook.append(output))

for k, v in unet1.named_modules():
    # attn1是self-attn, attn2是cross-attn
    # 这里注册的顺序不对，需要重新排序
    # 似乎不需要排序，forward放入list的tensor是顺序的
    if 'attn1.to_k' in k:
        print(f'register hook for {k}, {v}')
        v.register_forward_hook(to_k_forward)
    if 'attn1.to_v' in k:
        print(f'register hook for {k}, {v}')
        v.register_forward_hook(to_v_forward)

out = unet1(noisy_latents, timesteps, encoder_hidden_states)

print('to_k_hook', len(to_k_hook))
print('to_v_hook', len(to_v_hook))

for each in to_k_hook:
    print(each.shape)

# #################################################

global_k_idx = 0
global_v_idx = 0

def to_k2_hook(module, input_, output_):
    global global_k_idx
    assert output_.shape == to_v_hook[global_v_idx].shape
    res = torch.cat([output_, to_k_hook[global_k_idx]], dim=1)
    global_k_idx += 1
    return res

def to_v2_hook(module, input_, output_):
    # output_: [b, hw, c]
    global global_v_idx
    assert output_.shape == to_v_hook[global_v_idx].shape
    res = torch.cat([output_, to_v_hook[global_v_idx]], dim=1)
    global_v_idx += 1
    return res

for k, v in unet2.named_modules():
    # attn1是self-attn, attn2是cross-attn
    # 这里注册的顺序不对，需要重新排序
    if 'attn1.to_k' in k:
        print(f'register hook for {k}, {v}')
        v.register_forward_hook(to_k2_hook)
    if 'attn1.to_v' in k:
        print(f'register hook for {k}, {v}')
        v.register_forward_hook(to_v2_hook)


out = unet2(noisy_latents, timesteps, encoder_hidden_states)

print(out.sample.shape)
print(global_k_idx, global_v_idx)
print('done')

if __name__ == '__main__':
    pass