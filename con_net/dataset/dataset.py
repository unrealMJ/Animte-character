import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor

# Dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, control_type=None, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05):
        super().__init__()

        self.tokenizer = tokenizer
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate

        # self.data = json.load(open(json_file)) # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        data_root = '/mnt/petrelfs/majie/project/IP-Adapter/data/train'
        self.data = []
        for role in os.listdir(data_root):
            reference_path = f'{data_root}/{role}/image/0/0001.png'
            for idx in os.listdir(f'{data_root}/{role}/{control_type}'):
                for filename in os.listdir(f'{data_root}/{role}/{control_type}/{idx}'):
                    image_path = f'{data_root}/{role}/image/{idx}/{filename}'
                    caption_path = f'{data_root}/{role}/caption/{idx}/{filename.split(".")[0]}.txt'
                    control_path = f'{data_root}/{role}/{control_type}/{idx}/{filename}'

                    # TODO: load caption

                    self.data.append({
                        'image': image_path,
                        'caption': caption_path,
                        'control': control_path,
                        'reference': reference_path,
                    })


        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.control_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.clip_image_processor = CLIPImageProcessor()
        # self.clip_image_processor = CLIPImageProcessor(do_resize=True, size=384, do_center_crop=False)
        
    def __getitem__(self, idx):
        item = self.data[idx] 
        # text = item["caption"]
        text = ''
        image_file = item["image"]
        
        # read image
        raw_image = Image.open(image_file)
        image = self.transform(raw_image.convert("RGB"))

        clip_image = Image.open(item['reference']).convert("RGB")
        clip_image = self.clip_image_processor(images=clip_image, return_tensors="pt").pixel_values

        uncond = Image.open(item['reference']).convert("L")
        # uncond = np.asarray(uncond)
        # uncond = np.expand_dims(uncond, axis=0)
        # uncond = np.repeat(uncond, 3, axis=0)
        uncond = self.clip_image_processor(images=uncond, return_tensors="pt").pixel_values

        control = Image.open(item['control']).convert("RGB")
        control = self.control_transform(control)

        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1
        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
            "control": control,
            "uncond": uncond,
        }

    def __len__(self):
        return len(self.data)


class ValidateData:
    def __init__(self, control_type=None):
        self.data = {
            'train': [],
            'test': []
        }

        data_root = '/mnt/petrelfs/majie/project/IP-Adapter/data/test'
        for role in os.listdir(data_root):
            reference_path = f'{data_root}/{role}/image/0/0001.png'
            for idx in os.listdir(f'{data_root}/{role}/{control_type}'):
                for filename in os.listdir(f'{data_root}/{role}/{control_type}/{idx}'):
                    image_path = f'{data_root}/{role}/image/{idx}/{filename}'
                    caption_path = f'{data_root}/{role}/caption/{idx}/{filename.split(".")[0]}.txt'
                    control_path = f'{data_root}/{role}/{control_type}/{idx}/{filename}'

                    # TODO: load caption

                    self.data['test'].append({
                        'image': image_path,
                        'caption': caption_path,
                        'control': control_path,
                        'reference': reference_path,
                    })
        
        data_root = '/mnt/petrelfs/majie/project/IP-Adapter/data/train'
        for role in os.listdir(data_root):
            reference_path = f'{data_root}/{role}/image/0/0001.png'
            for idx in os.listdir(f'{data_root}/{role}/{control_type}'):
                for filename in os.listdir(f'{data_root}/{role}/{control_type}/{idx}'):
                    image_path = f'{data_root}/{role}/image/{idx}/{filename}'
                    caption_path = f'{data_root}/{role}/caption/{idx}/{filename.split(".")[0]}.txt'
                    control_path = f'{data_root}/{role}/{control_type}/{idx}/{filename}'

                    # TODO: load caption

                    self.data['train'].append({
                        'image': image_path,
                        'caption': caption_path,
                        'control': control_path,
                        'reference': reference_path,
                    })
