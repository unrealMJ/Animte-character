import torch
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from PIL import Image

import os

from transformers import CLIPTokenizer

from con_net.dataset.dataset2 import LaionHumanSD, CCTVDataset, BaseDataset
from con_net.utils import copy_src, image_grid

import torchvision.transforms as transforms
import einops
from inference.base_inferencer import BaseInferencer


class Inferencer(BaseInferencer):
    def __init__(self) -> None:
        super().__init__()
        pass

    def build_pipe(self):
        if hasattr(self.cfg, 'reference_path'):
            reference_path = self.cfg.reference_path
        else:
            reference_path = os.path.join(self.cfg.output_dir, f'checkpoints/checkpoint-{self.cfg.step}/pytorch_model.bin')
        
        reference_net = UNet2DConditionModel.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="unet")
        state_dict = torch.load(reference_path, map_location='cpu')
        reference_net.load_state_dict(state_dict)
        reference_net = reference_net.to(dtype=torch.float16, device='cuda')
        
        controlnet = ControlNetModel.from_pretrained(self.MODELS[self.cfg.control_type])
        controlnet = controlnet.to(dtype=torch.float16)
        
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
        tokenizer = CLIPTokenizer.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="tokenizer")
        dataset = BaseDataset(json_file=self.cfg.json_file, tokenizer=tokenizer, control_type=self.cfg.control_type)
        return dataset

    def validate(self):
        exp_dir = self.cfg.output_dir
        step = self.cfg.step
        for i in range(len(self.validate_data.data)):
            # if i > 10:
            #     break
            # j = random.randint(0, len(validate_data.data) - 1)
            j = i
            item = self.validate_data.data[j]

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


if __name__ == '__main__':
    # steps = list(range(0, 1000, 200))
    # steps = [25000, 20000]
    # for step in steps:
    #     infer = Inferencer()
    #     infer.build_pipe(step)
    #     infer.make_hook()
    #     infer.validate(step)

    # single image
    inferencer = Inferencer()
    inferencer.build_pipe()
    inferencer.make_hook()
    prompt = '1girl,upper body,cry,sad'

    reference_path = '/mnt/petrelfs/majie/project/My-IP-Adapter/data/test_demo/0.png'
    control_path = '/mnt/petrelfs/majie/project/My-IP-Adapter/data/test_demo/pose2.png'

    reference = Image.open(reference_path).convert("RGB")
    control_image = Image.open(control_path).convert("RGB")

    grid = inferencer.single_image_infer(reference, prompt, control_image)
    inferencer.save(grid, prompt, inferencer.cfg.output_dir, inferencer.cfg.step)
