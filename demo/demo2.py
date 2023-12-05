import os
import shutil


root = '/mnt/petrelfs/majie/datasets/TikTok/TikTok_dataset'
for role in os.listdir(root):
    pose_root = os.path.join(root, role, 'pose')
    tmp = os.listdir(pose_root)[0]
    tmp_root = os.path.join(pose_root, tmp)

    if tmp.endswith('png'):
        print(f'already done {pose_root}')
        continue


    for each in os.listdir(tmp_root):
        src_path = os.path.join(tmp_root, each)
        dst_path = os.path.join(pose_root, each)
        shutil.move(src_path, dst_path)
    os.rmdir(tmp_root)
    print(pose_root)