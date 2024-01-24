import cv2
import os
from PIL import Image
from mmpose.apis import MMPoseInferencer
from tqdm import tqdm
import multiprocessing
import concurrent.futures
from tqdm import tqdm

pose2d='/mnt/petrelfs/majie/project/openmmlab/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288.py'
pose2d_weights='https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288-70d7ab01_20220913.pth'

# 使用模型别名创建推断器
inferencer = MMPoseInferencer(
    pose2d=pose2d,
    pose2d_weights=pose2d_weights,
    show_progress=True
)

def generate_single_pose(video_path):
    file_dir = video_path.split('/')[-2]
    file_name = video_path.split('/')[-1]
    
    output_root = '/mnt/petrelfs/share_data/majie/douyin/pose'
    output_dir = os.path.join(output_root, file_dir)
    target_file = os.path.join(output_dir, file_name)
    
    if video_path.endswith('jpeg'):
        return

    if os.path.exists(target_file):
        src_video = cv2.VideoCapture(video_path)
        pose_video = cv2.VideoCapture(target_file)

        src_frame_count = int(src_video.get(cv2.CAP_PROP_FRAME_COUNT))
        pose_frame_count = int(pose_video.get(cv2.CAP_PROP_FRAME_COUNT))

        src_video.release()
        pose_video.release()

        if src_frame_count == pose_frame_count:
            return
        else:
            os.remove(target_file)
            os.remove(target_file.replace('.mp4', '.json'))
    
    # inferencer = MMPoseInferencer(
    #     pose2d=pose2d,
    #     pose2d_weights=pose2d_weights,
    #     show_progress=True
    # )
    result_generator = inferencer(video_path, pred_out_dir=output_dir, vis_out_dir=output_dir, kpt_thr=0.5, thickness=3, radius=5,
                                  skeleton_style='openpose', black_background=True)
    
    for each in result_generator:
        # print(target_file)
        # print(f'type {type(each)}')
        result = each


if __name__ == '__main__':
    root_path = '/mnt/petrelfs/share_data/majie/douyin/video'
    all_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            all_files.append(os.path.join(root, file))

    # with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
    #     executor.map(generate_single_pose, all_files)
    
    # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    #     executor.map(generate_single_pose, all_files)
    
    # generate_single_pose(all_files[3])
    for each in tqdm(all_files):
        generate_single_pose(each)