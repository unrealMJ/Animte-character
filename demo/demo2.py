import os
import shutil
import jsonlines


def generate_video_jsonl():
    data = []
    root = '/mnt/petrelfs/majie/project/diffusers/my_project/identity/TikTok/TikTok_dataset'
    for role in os.listdir(root):
        item = {}
        item['role'] = role
        item['role_root'] = os.path.join(root, role)
        item['prompt'] = ''

        data.append(item)
    
    with jsonlines.open('video.jsonl', 'w') as writer:
        writer.write_all(data)

generate_video_jsonl()