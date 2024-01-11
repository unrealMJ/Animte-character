import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
import PIL
from PIL import Image
from transformers import CLIPImageProcessor
import json
from torch.utils.data import IterableDataset, Dataset
from tqdm import tqdm
import jsonlines
import copy
from controlnet_aux import HEDdetector, OpenposeDetector
import cv2


class TikTok(Dataset):
    def __init__(self, data_root, images_file, tokenizer, size=(512, 512), t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, use_control=False, control_type='canny'):
        super().__init__()
        self.data_root = data_root
        data = open(images_file).readlines()
        self.data = self.construct_data(data)
        print(f'Dataset size: {len(self.data)}')

        self.tokenizer = tokenizer
        self.size = size
        self.t_drop_rate = t_drop_rate
        self.i_drop_rate = i_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.use_control = use_control
        self.control_type = control_type
        if self.control_type == 'hed':
            self.hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
        elif self.control_type == 'pose':
            self.pose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

        # TODO: support ARB bucket
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def construct_data(self, data):
        filter_data = []
        for i in tqdm(range(len(data))):
            each = {}
            each['img_path'] = data[i].strip()
            filter_data.append(each)
        return filter_data

    def control_transform(self, control_image):
        control_image = transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR)(control_image)
        control_image = transforms.CenterCrop(self.size)(control_image)
        control_image = transforms.ToTensor()(control_image)
        return control_image

    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            image_file = item["img_path"]
            raw_image = Image.open(image_file).convert("RGB")
            image_base_path = os.path.dirname(image_file)

            frame_files = os.listdir(image_base_path)
            frame_files = [os.path.join(image_base_path, f) for f in frame_files]
            frame_files.sort()
            IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
            all_images = []
            for frame in frame_files:
                frame_extension = os.path.splitext(frame)[1]
                if frame_extension in IMAGE_EXTENSIONS:
                    all_images.append(frame)
            
            reference_image_ind = random.randint(0, len(all_images) - 1)
            reference_image = Image.open(all_images[reference_image_ind]).convert("RGB")

            prompt = ""
            image = self.transform(raw_image)
            reference = self.transform(reference_image)
            control_image = None
            if self.use_control:
                if self.control_type == 'canny':
                    control_image = np.array(raw_image)
                    control_image = cv2.Canny(control_image, 100, 200)
                    control_image = control_image[:, :, None]
                    control_image = np.concatenate([control_image, control_image, control_image], axis=2)
                    control_image = Image.fromarray(control_image)
                elif self.control_type == 'hed':
                    control_image = self.hed(raw_image)
                elif self.control_type == 'pose':
                    control_image = self.pose(raw_image, hand_and_face=True)

                control_image = self.control_transform(control_image)

        except Exception:
            return self.__getitem__((idx + 1) % len(self.data))

        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            prompt = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            prompt = ""
            drop_image_embed = 1

        text_input_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        # reference net 的 prompt可以是空文本？
        return {
            'image': image,
            'text_input_ids': text_input_ids,
            'reference': reference,
            "drop_image_embed": drop_image_embed,
            'control_image': control_image,
        }

    def __len__(self):
        return len(self.data)
