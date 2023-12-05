import torch
from safetensors import safe_open

from convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint, convert_lora


def save_img(img):
    import time
    # format time string
    timestr = time.strftime("%Y%m%d-%H%M%S")
    img.save(f'output/{timestr}.png')


def load_db(pipeline, model_path, lora_path=None, lora_alpha=1.0):
    if model_path.endswith(".ckpt"):
        state_dict = torch.load(model_path)
        pass 
    elif model_path.endswith(".safetensors"):
        state_dict = {}
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
                
        is_lora = all("lora" in k for k in state_dict.keys())
        if not is_lora:
            base_state_dict = state_dict
        else:
            # base_state_dict = {}
            # with safe_open(model_config.base, framework="pt", device="cpu") as f:
            #     for key in f.keys():
            #         base_state_dict[key] = f.get_tensor(key) 
            pass       
        
        # vae
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, pipeline.vae.config)
        pipeline.vae.load_state_dict(converted_vae_checkpoint, strict=True)
        # unet
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, pipeline.unet.config)
        pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=True)
        # text_model
        # 加载权重会导致text encoder由16位转为32位，原因是自己加载了新的text encoder
        # 直接传入16位的text encoder，在此基础上加载权重
        pipeline.text_encoder = convert_ldm_clip_checkpoint(base_state_dict, text_encoder=pipeline.text_encoder)
        
        # import pdb
        # pdb.set_trace()

        if lora_path is not None:
            lora_dict = {}
            with safe_open(lora_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    lora_dict[key] = f.get_tensor(key)

            pipeline = convert_lora(pipeline, lora_dict, alpha=lora_alpha)
    return pipeline

