from transformers import CLIPImageProcessor
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPVisionModel
from PIL import Image
import torch.nn as nn
import torch


if __name__ == '__main__':
    image_encoder = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')
    # image_encoder = CLIPVisionModel.from_pretrained('/mnt/petrelfs/majie/project/IP-Adapter/IP-Adapter/models/image_encoder')
    clip_image_processor = CLIPImageProcessor()

    image = Image.open('/mnt/petrelfs/majie/project/IP-Adapter/data/train/role_0/image/0/0001.png')
    image = clip_image_processor(image, return_tensors="pt").pixel_values

    print(image.shape)  # [1, 3, height, width]

    fea = image_encoder(image)
    print(type(fea))
    print(fea.pooler_output.shape)
    print(fea.last_hidden_state.shape)
    