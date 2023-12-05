import json
import os
import jsonlines


def build_tiktok_jsonl():
    data_root = '/mnt/petrelfs/majie/datasets/TikTok/TikTok_dataset'

    data = []
    for role in os.listdir(data_root):
        image_root = os.path.join(data_root, role, 'images')
        pose_root = os.path.join(data_root, role, 'pose')

        for i, each in enumerate(os.listdir(image_root)):
            if i % 20 != 0:
                continue
            d = {}
            d['role'] = role
            d['image'] = os.path.join(image_root, each)
            d['pose'] = os.path.join(pose_root, each)
            d['reference'] = os.path.join(image_root, f'0001.png')
            data.append(d)

    with jsonlines.open('tiktok_all.jsonl', 'w') as writer:
        for each in data:
            writer.write(each)

def build_cctv_jsonl():
    data = []
    data_root = '/mnt/petrelfs/majie/project/diffusers/my_project/identity/data'

    image_root = f'{data_root}/resize_frame'
    pose_root = f'{data_root}/pose'
    canny_root = f'{data_root}/canny'

    for role in os.listdir(image_root):
        for image in os.listdir(f'{image_root}/{role}'):
            d = {}
            d['role'] = role
            d['image'] = f'{image_root}/{role}/{image}'
            d['pose'] = f'{pose_root}/{role}/{image}'
            d['canny'] = f'{canny_root}/{role}/{image}'
            d['prompt'] = ''
            d['reference'] = f'{image_root}/{role}/frame_0.png'
            
            data.append(d)
    
    with jsonlines.open('cctv_all.jsonl', 'w') as writer:
        for each in data:
            writer.write(each)



if __name__ == '__main__':
    build_cctv_jsonl()