import jsonlines
import os


new_data = []
root = '/mnt/petrelfs/majie/project/diffusers/my_project/identity/data2'

image_root = os.path.join(root, 'image')
canny_root = os.path.join(root, 'canny')
pose_root = os.path.join(root, 'pose')


for role in os.listdir(image_root):
    role_path = os.path.join(image_root, role)
    all_files = os.listdir(role_path)
    if role == '00049' or role == '00050':
        continue
    for i in range(len(all_files)):  # 第i张图为参考
        for j in range(len(all_files)):
            d = {
                'role': role,
                'image': os.path.join(role_path, all_files[j]),
                'canny': os.path.join(canny_root, role, all_files[j]),
                'pose': os.path.join(pose_root, role, all_files[j]),
                'reference': os.path.join(role_path, all_files[i]),
                'prompt': ''
            }

            new_data.append(d)

out_path = '/mnt/petrelfs/majie/project/My-IP-Adapter/data/cctv/aug/sixpose.jsonl'
with jsonlines.open(out_path, 'w') as writer:
    writer.write_all(new_data)