import gradio as gr
from PIL import Image
from diffusers import (StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel, StableDiffusionPipeline)
from ip_adapter import IPAdapter
from ip_adapter.utils import generate
import torch
import yaml
import os
from types import MethodType
import time


class GradioPipeline:
    def __init__(self):
        self.config = 'config/base.yaml'
        self.image_encoder_path = '/mnt/petrelfs/majie/project/IP-Adapter/IP-Adapter/models/image_encoder'
        self.current_sd = None
        self.current_ip = None
        self.current_sd_control = None

        self.sd_root = '/mnt/petrelfs/majie/model_checkpoint'

        self.set_sd('stable-diffusion-v1-5')
        self.set_ip('2023-11-16-10-15', 48000)

    def set_sd(self, sd_checkpoint):
        if self.current_sd == sd_checkpoint:
            return
        self.current_sd = sd_checkpoint
        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        sd_checkpoint = os.path.join(self.sd_root, sd_checkpoint)
        if sd_checkpoint.endswith('.safetensors') or sd_checkpoint.endswith('.ckpt'):
            ip_pipe = StableDiffusionPipeline.from_single_file(
                sd_checkpoint,
                torch_dtype=torch.float16,
                feature_extractor=None,
                safety_checker=None,
                requires_safety_checker=False
            )
        else:
            vae_model_path = "/mnt/petrelfs/majie/model_checkpoint/sd-vae-ft-mse"
            infer_vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
            
            ip_pipe = StableDiffusionPipeline.from_pretrained(
                sd_checkpoint,
                torch_dtype=torch.float16,
                scheduler=ddim_scheduler,
                vae=infer_vae,
                feature_extractor=None,
                safety_checker=None
            )

        ip_pipe.__call__ = MethodType(generate, ip_pipe)

        ip_adapter_inference = IPAdapter(sd_pipe=ip_pipe, image_encoder_path=self.image_encoder_path,
                            ip_ckpt=None, device='cuda')

        self.ip_adapter_inference = ip_adapter_inference
        print(f'Load SD-Checkpoint: {self.current_sd}')
        
        # tmp = self.current_ip
        # self.current_ip = None
        # self.set_ip(tmp)
    
    def set_ip(self, ip_adapter_path, step):
        ip_adapter_path = os.path.join('output/exp1', ip_adapter_path, f'checkpoint-{step}')
        if self.current_ip == ip_adapter_path:
            return
        self.current_ip = ip_adapter_path
        state_dict = torch.load(f'{self.current_ip}/pytorch_model.bin', map_location='cpu')
        image_proj_state_dict = {}
        ip_state_dict = {}
        clip_state_dict = {}
        for k, v in state_dict.items():
            if 'image_proj_model.' in k:
                image_proj_state_dict[k.replace('image_proj_model.', '')] = v
            elif 'adapter_modules.' in k:
                ip_state_dict[k.replace('adapter_modules.', '')] = v
            elif 'image_encoder.' in k:
                clip_state_dict[k.replace('image_encoder.', '')] = v
            else:
                pass
        if len(clip_state_dict) > 0:
            self.ip_adapter_inference.image_encoder.load_state_dict(clip_state_dict, strict=False)
        self.ip_adapter_inference.image_proj_model.load_state_dict(image_proj_state_dict)
        ip_layers = torch.nn.ModuleList(self.ip_adapter_inference.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(ip_state_dict)
        print(f'Load IP-Adapter weights: {self.current_ip}')

    def set_sd_control(self, control_checkpoint):
        if self.current_sd_control == control_checkpoint:
            return
        self.current_sd_control = control_checkpoint

        vae_model_path = "/mnt/petrelfs/majie/model_checkpoint/sd-vae-ft-mse"
        infer_vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        ip_control_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            control_checkpoint,
            torch_dtype=torch.float16,
            scheduler=ddim_scheduler,
            vae=infer_vae,
            feature_extractor=None,
            safety_checker=None
        )
        ip_control_pipe.__call__ = MethodType(generate, ip_control_pipe)
        self.ip_adapter_control = IPAdapter(sd_pipe=ip_control_pipe, image_encoder_path=self.image_encoder_path, ip_ckpt=None, device='cuda')

    def generate(self, batch_size=4, prompt=None, reference_image=None, control_map=None, enable_controlnet=False):
        print(f'Generate')
        results = self.single_image_infer(batch_size=batch_size, prompt=prompt, reference_image=reference_image, control_map=control_map)
        self.save(results)
        return results

    def single_image_infer(self, batch_size=4, prompt=None, reference_image=None, control_map=None):
        width, height = reference_image.size
        width = (width // 8) * 8
        height = (height // 8) * 8
        reference_image = reference_image.resize((width, height), Image.BILINEAR)

        ip_result = self.ip_adapter_inference.generate(pil_image=reference_image, prompt=prompt, num_samples=batch_size, width=width,
                                    height=height, num_inference_steps=50)

        all_images = [reference_image] + ip_result
        return all_images

    def save(self, results):
        current_time = time.strftime("%Y-%m-%d")
        os.makedirs(f'ui_output/{current_time}', exist_ok=True)
        for i, each in enumerate(results):
            each.save(f'ui_output/{current_time}/{time.strftime("%Y-%m-%d-%H-%M-%S")}_{i}.png')


pipe = GradioPipeline()

sd_list = ['anythingV3_fp16.ckpt', 'animesfw-final-pruned.ckpt', 'Realistic_Vision_V4.0_fp16-no-ema.safetensors', 'stable-diffusion-v1-5']
ip_weight_list = ['2023-11-16-10-15', '2023-11-18-09-37', '2023-11-21-11-32', '2023-11-21-12-14']

with gr.Blocks() as demo:
    with gr.Row():
        sd_checkpoint = gr.Dropdown(label="Checkpoint", choices=sd_list, value=pipe.current_sd)
        ip_adapter_path = gr.Dropdown(label="IP path", choices=ip_weight_list)
        step = gr.Dropdown(label="Step", choices=list(range(0, 50000, 2000)))
        load_btn = gr.Button("Load")
    with gr.Row():
        batch_size = gr.Number(label="Batch Size", minimum=1, maximum=8, precision=0, value=4)
        enable_controlnet = gr.Checkbox(label="Enable ControlNet")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", lines=3, value='best quality,high quality')
        generate_btn = gr.Button("Generate")
    with gr.Row():
        reference_image = gr.Image(label="Reference Image", type='pil')
        control_map = gr.Image(label="Control Map", type='pil')

    output = gr.Gallery(label="Output", columns=6)

    generate_btn.click(
        fn=pipe.generate,
        inputs=[
            batch_size,
            prompt,
            reference_image,
            control_map
        ],
        outputs=output
    )
    sd_checkpoint.change(
        fn=pipe.set_sd,
        inputs=[sd_checkpoint]
    )
    # ip_adapter_path.change(
    #     fn=pipe.set_ip,
    #     inputs=[ip_adapter_path]
    # )
    load_btn.click(
        fn=pipe.set_ip,
        inputs=[ip_adapter_path, step],
        outputs=None
    )


demo.launch(server_name='0.0.0.0', server_port=12138, allowed_paths=['*'], share=False)