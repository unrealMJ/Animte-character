import os
import torch
import torch.nn as nn
import torch.nn.init as init
from einops import rearrange
import numpy as np
from torch.nn import functional as F
from typing import Any, Dict, List, Optional, Tuple, Union


class PoseGuider(nn.Module):
    def __init__(self, noise_latent_channels=4):
        super(PoseGuider, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Final projection layer
        self.final_proj = nn.Conv2d(in_channels=128, out_channels=noise_latent_channels, kernel_size=1)

        # Initialize layers
        self._initialize_weights()

        self.scale = nn.Parameter(torch.ones(1))

    # def _initialize_weights(self):
    #     # Initialize weights with Gaussian distribution and zero out the final layer
    #     for m in self.conv_layers:
    #         if isinstance(m, nn.Conv2d):
    #             init.normal_(m.weight, mean=0.0, std=0.02)
    #             if m.bias is not None:
    #                 init.zeros_(m.bias)

    #     init.zeros_(self.final_proj.weight)
    #     if self.final_proj.bias is not None:
    #         init.zeros_(self.final_proj.bias)
    
    def _initialize_weights(self):
        # Initialize weights with He initialization and zero out the biases
        for m in self.conv_layers:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                init.normal_(m.weight, mean=0.0, std=np.sqrt(2. / n))
                if m.bias is not None:
                    init.zeros_(m.bias)

        # For the final projection layer, initialize weights to zero (or you may choose to use He initialization here as well)
        init.zeros_(self.final_proj.weight)
        if self.final_proj.bias is not None:
            init.zeros_(self.final_proj.bias)


    def forward(self, x):
        x = self.conv_layers(x)
        x = self.final_proj(x)

        return x * self.scale

    @classmethod
    def from_pretrained(cls,pretrained_model_path):
        if not os.path.exists(pretrained_model_path):
            print(f"There is no model file in {pretrained_model_path}")
        print(f"loaded PoseGuider's pretrained weights from {pretrained_model_path} ...")

        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        model = PoseGuider(noise_latent_channels=4)
                
        m, u = model.load_state_dict(state_dict, strict=False)
        # print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")        
        params = [p.numel() for n, p in model.named_parameters()]
        print(f"### PoseGuider's Parameters: {sum(params) / 1e6} M")
        
        return model


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int = 4,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
