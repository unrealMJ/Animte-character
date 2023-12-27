from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection
from PIL import Image
import requests
import torch.nn as nn
import torch

class DINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino = AutoModel.from_pretrained('facebook/dinov2-base')
        hidden_size = self.dino.config.hidden_size
        self.linear = nn.Linear(hidden_size, 768)

    def forward(self, x):
        # preprocess the image in dataset.py
        x = self.dino(x).last_hidden_state  # 16bit
        # print(f'0 x.shape: {x.shape} dtype: {x.dtype}')
        # x.to(dtype=torch.float16)
        x = self.linear(x)  # 32bit
        # print(f'1 x.shape: {x.shape} dtype: {x.dtype}')
        return x


class CLIPImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14')
        
        self.linear = nn.Linear(self.clip.config.hidden_size, 768)
    
    def forward(self, x):
        x = self.clip(x).last_hidden_state

        x = self.linear(x)

        return x

if __name__ == '__main__':
    model = DINO().to('cuda')
    model.dino.to(dtype=torch.float16)

    images = torch.randn(1, 3, 224, 224)
    images = images.to(device='cuda', dtype=torch.float16)

    out = model(images)

    print(f'out.shape: {out.shape} dtype: {out.dtype}')
