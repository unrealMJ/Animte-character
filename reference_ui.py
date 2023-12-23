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
from inference.validate_reference import Inferencer


class GradioPipeline:
    def __init__(self):
        # self.inferencer = Inferencer()
        
        # self.inferencer.build_pipe()
        # self.inferencer.make_hook()
        pass


    def set_sd(self, sd_checkpoint):
        raise NotImplementedError

    def set_reference(self, reference):
        raise NotImplementedError

    def set_controlnet(self, controlnet):
        raise NotImplementedError

    def generate(self, reference, prompt, control_image):
        print(f'Generate')
        return [reference]
        all_images = self.inferencer.single_image_infer(reference, prompt, control_image, return_raw=True)
        self.save(all_images)
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
        sd_checkpoint = gr.Dropdown(label="Checkpoint", choices=sd_list)
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
        reference = gr.Image(label="Reference Image", type='pil')
        control_iamge = gr.Image(label="Control Image", type='pil')

    output = gr.Gallery(label="Output", columns=6)

    generate_btn.click(
        fn=pipe.generate,
        inputs=[
            prompt,
            reference,
            control_iamge
        ],
        outputs=output
    )


demo.launch(server_name='0.0.0.0', server_port=15000, allowed_paths=['*'], share=False)

# run this script with 
'''
proxy_off

python reference_ui.py --config config/inference/reference_net.yaml output_dir=output/reference_net/2023-12-04-17-27 step=30000

'''
