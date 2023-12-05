import os
import random
import argparse
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from diffusers import (
    AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel, StableDiffusionControlNetPipeline,
    DDIMScheduler
    )
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPVisionModel

import cv2
import numpy as np



# TODO: support reset and log

class Metric:
    def __init__(self, device, image_encoder_path='/mnt/petrelfs/majie/model_checkpoint/clip-vit-large-patch14'):
        self.device = device
        self.image_encoder = CLIPVisionModel.from_pretrained(image_encoder_path).to(self.device, dtype=torch.float16)
        # self.image_encoder = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32').to(self.device, dtype=torch.float16)
        self.clip_image_processor = CLIPImageProcessor()

    @torch.no_grad()
    def clip_sim(self, results, gt):
        # TODO: debug
        sim = []
        gt = self.clip_image_processor(gt, return_tensors="pt").pixel_values
        gt = self.image_encoder(gt.to(self.device, dtype=torch.float16)).pooler_output  # [1, 768]
        for result in results:
            result = self.clip_image_processor(result, return_tensors="pt").pixel_values
            result = self.image_encoder(result.to(self.device, dtype=torch.float16)).pooler_output
            sim.append(torch.cosine_similarity(result, gt, dim=-1).item())

        # results = self.clip_image_processor(results).pixel_values
        # results = self.image_encoder(results).pooler_output
        return sum(sim) / len(sim)
    
    def psnr(self, results, gt):
        # https://mmagic.readthedocs.io/zh-cn/latest/user_guides/metrics.html#psnr
        # /mnt/petrelfs/majie/project/mmagic/mmagic/evaluation/metrics/psnr.py
        results = [np.array(each) for each in results]
        gt = np.array(gt)

        tmp = []
        for each in results:
            mse_value = ((each - gt)**2).mean()
            if mse_value == 0:
                result = float('inf')
            else:
                result = 20. * np.log10(255. / np.sqrt(mse_value))

            tmp.append(result)
        
        return sum(tmp) / len(tmp)
    
    def l1_error(self, results, gt):
        # /mnt/petrelfs/majie/project/mmagic/mmagic/evaluation/metrics/mae.py
        results = [np.array(each) / 255. for each in results]
        gt = np.array(gt) / 255.

        tmp = []
        for each in results:
            result = np.abs(each - gt).mean()
            tmp.append(result)
        
        return sum(tmp) / len(tmp)
    
    def ssim(self, results, gt):
        # /mnt/petrelfs/majie/project/mmagic/mmagic/evaluation/metrics/ssim.py
        results = [np.array(each) for each in results]
        gt = np.array(gt)

        tmp = []
        for each in results:
            for i in range(gt.shape[2]):
                result = _ssim(each[:, :, i], gt[:, :, i])
                tmp.append(result)

        return np.array(tmp).mean()


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`ssim`.

    Args:
        img1, img2 (np.ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: SSIM result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()



if __name__ == '__main__':
    img1 = Image.open('data/1.jpg')
    img2 = Image.open('data/1.jpg')

    img2 = [img2]
    metric = Metric('cuda')
    print(metric.clip_sim(img2, img1))
