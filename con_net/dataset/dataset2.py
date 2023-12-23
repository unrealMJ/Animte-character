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
from transformers import CLIPImageProcessor, AutoImageProcessor
import json
from torch.utils.data import IterableDataset, Dataset
from tqdm import tqdm
import jsonlines
import copy


class LaionHumanSD(Dataset):
    def __init__(self, json_file, tokenizer, size=(512, 512), t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05):
        super().__init__()
        self.data_root = '/mnt/petrelfs/majie/project/HumanSD/humansd_data/datasets/Laion'
        with open(json_file) as f:
            data = json.load(f)
        self.data = self.construct_data(data)
        print(f'Dataset size: {len(self.data)}')

        self.tokenizer = tokenizer
        self.size = size
        self.t_drop_rate = t_drop_rate
        self.i_drop_rate = i_drop_rate
        self.ti_drop_rate = ti_drop_rate

        # TODO: support ARB bucket
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def construct_data(self, data):
        keys = list(data.keys())
        values = list(data.values())
        filter_data = []
        for i in tqdm(range(len(values))):
            each = values[i]
            each['key'] = keys[i]
            each['img_path'] = os.path.join(self.data_root, each['img_path'])
            filter_data.append(each)
        return filter_data

    def remove_not_exist(self, data):
        keys = list(data.keys())
        values = list(data.values())
        filter_data = []
        for i in tqdm(range(len(values))):
            each = values[i]
            each['key'] = keys[i]
            each['img_path'] = os.path.join(self.data_root, each['img_path'])
            if os.path.exists(each['img_path']):
                filter_data.append(each)
        return filter_data
        
    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            image_file = item["img_path"]
            raw_image = Image.open(image_file).convert("RGB")
            prompt = item["prompt"]
            image = self.transform(raw_image)
            reference = self.transform(raw_image)
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
        }

    def __len__(self):
        return len(self.data)


class BaseDataset(Dataset):
    def __init__(self, json_file, tokenizer=None, short_size=768, control_type='pose') -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.short_size = short_size
        self.control_type = control_type

        self.data = self.construct_data(json_file)

    def construct_data(self, json_file_list):
        if type(json_file_list) == str:
            json_file_list = [json_file_list]

        data = []
        for json_file in json_file_list:
            with jsonlines.open(json_file) as reader:
                for each in reader:
                    data.append(each)
        return data
    
    def image_transform(self, image):
        # image = transforms.Resize(self.short_size, interpolation=transforms.InterpolationMode.BILINEAR)(image)
        # image = transforms.CenterCrop(self.short_size)(image)

        # i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(512, 512))
        # image = TF.crop(image, i, j, h, w)

        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.5], [0.5])(image)

        return image
    
    def reference_transform(self, reference):
        # reference = transforms.Resize(self.short_size, interpolation=transforms.InterpolationMode.BILINEAR)(reference)
        # reference = transforms.CenterCrop(self.short_size)(reference)

        reference = transforms.ToTensor()(reference)
        reference = transforms.Normalize([0.5], [0.5])(reference)

        return reference

    def control_transform(self, control_image):
        # control_image = transforms.Resize(self.short_size, interpolation=transforms.InterpolationMode.BILINEAR)(control_image)
        # control_image = transforms.CenterCrop(self.short_size)(control_image)
        control_image = transforms.ToTensor()(control_image)
        return control_image

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image']).convert("RGB")
        reference = Image.open(item['reference']).convert("RGB")
        
        if self.control_type == 'canny':
            control_image = Image.open(item['canny']).convert("RGB")
        elif self.control_type == 'pose':
            control_image = Image.open(item['pose']).convert("RGB")
        else:
            raise NotImplementedError

        if 'prompt' not in item or item['prompt'] == '':
            prompt = 'best quality,high quality'
        else:
            prompt = item['prompt']
        
        width, height = image.size
        width = (width // 8) * 8
        height = (height // 8) * 8

        image = image.resize((width, height))
        reference = reference.resize((width, height))
        control_image = control_image.resize((width, height))

        image = self.image_transform(image)
        reference = self.reference_transform(reference)
        control_image = self.control_transform(control_image)

        text_input_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            'image': image,
            'text_input_ids': text_input_ids,
            'reference': reference,
            'control_image': control_image,
        }

    def __len__(self) -> int:
        return len(self.data)


class CCTVDataset(BaseDataset):
    pass


class TikTokDataset(BaseDataset):
    def __init__(self, json_file, tokenizer, short_size=768, control_type='pose') -> None:
        super().__init__(json_file, tokenizer, short_size, control_type)

    def __getitem__(self, idx):
        item = self.data[idx]

        role_root = item['role_root']
        all_images = os.listdir(f'{role_root}/images')
        video_length = len(all_images)
        
        reference_idx = random.randint(0, video_length - 1)
        image_idx = random.randint(0, video_length - 1)

        reference_path = f'{role_root}/images/{str(reference_idx + 1).zfill(4)}.png'
        image_path = f'{role_root}/images/{str(image_idx + 1).zfill(4)}.png'
        control_path = f'{role_root}/{self.control_type}/{str(image_idx + 1).zfill(4)}.png'

        reference = Image.open(reference_path).convert("RGB")
        image = Image.open(image_path).convert("RGB")
        control_image = Image.open(control_path).convert("RGB")

        reference_prompt = ''
        # with open(f'{role_root}/prompt/{str(image_idx + 1).zfill(4)}.txt') as f:
        #     prompt = f.read()
        prompt = 'best quality,high quality'

        reference = self.reference_transform(reference)
        image = self.image_transform(image)
        control_image = self.control_transform(control_image)

        reference_prompt = self.tokenizer(
            reference_prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        prompt = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            'image': image,
            'text_input_ids': prompt,
            'reference': reference,
            'control_image': control_image,
            'reference_prompt': reference_prompt,
        }


class TikTokDataset2(BaseDataset):
    '''
    This class not included text prompt
    '''
    def __init__(self, json_file, tokenizer=None, short_size=768, control_type='pose', processor=None) -> None:
        super().__init__(json_file, tokenizer, short_size, control_type)
        self.processor = processor

    def __getitem__(self, idx):
        item = self.data[idx]

        role_root = item['role_root']
        all_images = os.listdir(f'{role_root}/images')
        video_length = len(all_images)
        
        reference_idx = random.randint(0, video_length - 1)
        image_idx = random.randint(0, video_length - 1)

        reference_path = f'{role_root}/images/{str(reference_idx + 1).zfill(4)}.png'
        image_path = f'{role_root}/images/{str(image_idx + 1).zfill(4)}.png'
        control_path = f'{role_root}/{self.control_type}/{str(image_idx + 1).zfill(4)}.png'

        reference = Image.open(reference_path).convert("RGB")
        image = Image.open(image_path).convert("RGB")
        control_image = Image.open(control_path).convert("RGB")

        width, height = image.size
        width = (width // 8) * 8
        height = (height // 8) * 8

        image = image.resize((width, height))
        reference = reference.resize((width, height))
        control_image = control_image.resize((width, height))
        
        global_image = self.processor(images=reference, return_tensors="pt").pixel_values

        reference = self.reference_transform(reference)
        image = self.image_transform(image)
        control_image = self.control_transform(control_image)
        
        return {
            'image': image,
            'global_image': global_image,
            'reference': reference,
            'control_image': control_image,
        }
        

class UBCFashion(TikTokDataset):
    pass
    

class BaseVideoDataset(Dataset):
    def __init__(self, json_file, tokenizer, control_type='canny', sample_size=512, sample_stride=4, sample_n_frames=24) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.short_size = sample_size  # maybe not used
        self.control_type = control_type
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames

        self.data = self.construct_data(json_file)

    def construct_data(self, json_file_list):
        if type(json_file_list) == str:
            json_file_list = [json_file_list]

        data = []
        for json_file in json_file_list:
            with jsonlines.open(json_file) as reader:
                for each in reader:
                    data.append(each)
        return data
    
    def image_transform(self, image):
        image = transforms.Resize(self.short_size, interpolation=transforms.InterpolationMode.BILINEAR)(image)
        image = transforms.CenterCrop(self.short_size)(image)

        # i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(512, 512))
        # image = TF.crop(image, i, j, h, w)

        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.5], [0.5])(image)

        return image
    
    def reference_transform(self, reference):
        reference = transforms.Resize(self.short_size, interpolation=transforms.InterpolationMode.BILINEAR)(reference)
        reference = transforms.CenterCrop(self.short_size)(reference)

        # i, j, h, w = transforms.RandomCrop.get_params(reference, output_size=(512, 512))
        # reference = TF.crop(reference, i, j, h, w)
        
        reference = transforms.ToTensor()(reference)
        reference = transforms.Normalize([0.5], [0.5])(reference)

        return reference

    def control_transform(self, control_image):
        control_image = transforms.Resize(self.short_size, interpolation=transforms.InterpolationMode.BILINEAR)(control_image)
        control_image = transforms.CenterCrop(self.short_size)(control_image)
        control_image = transforms.ToTensor()(control_image)
        return control_image

    def __getitem__(self, idx):
        item = self.data[idx]

        role_root = item['role_root']
        all_images = os.listdir(f'{role_root}/images')
        video_length = len(all_images)

        clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)

        image_list = []
        control_list = []
        for idx in batch_index:
            idx = str(idx + 1).zfill(4)
            image = Image.open(f'{role_root}/images/{idx}.png').convert("RGB")
            image_list.append(image)
            control_image = Image.open(f'{role_root}/{self.control_type}/{idx}.png').convert("RGB")
            control_list.append(control_image)

        reference_path = f'{role_root}/images/0001.png'
        reference = Image.open(reference_path).convert("RGB")
        
        if 'prompt' not in item or item['prompt'] == '':
            prompt = 'best quality,high quality'
        else:
            prompt = item['prompt']
        
        width, height = reference.size
        width = (width // 8) * 8
        height = (height // 8) * 8

        reference = reference.resize((width, height))
        image_list = [image.resize((width, height)) for image in image_list]
        control_list = [control_image.resize((width, height)) for control_image in control_list]

        # TODO: support random crop, keep consistency between image and control
        reference = self.reference_transform(reference)
        image_list = [self.image_transform(image) for image in image_list]
        control_list = [self.control_transform(control_image) for control_image in control_list]

        text_input_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        video = torch.stack(image_list, dim=0)  # [n_frames, 3, h, w]
        control_video = torch.stack(control_list, dim=0)

        return {
            'video': video,
            'text_input_ids': text_input_ids,
            'reference': reference,
            'control_video': control_video,
        }

    def __len__(self) -> int:
        return len(self.data)





if __name__ == '__main__':
    dataset = LaionHumanSD('/mnt/petrelfs/majie/project/HumanSD/humansd_data/datasets/Laion/Aesthetics_Human/mapping_file_training.json', None)
