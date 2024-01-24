import cv2
import os
from PIL import Image
from mmpose.apis import MMPoseInferencer
from tqdm import tqdm


def extract_single_video(video_path, video_num, save_interval=1, output_dir='path/to/output'):
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(video_path)
        print("无法打开视频文件")
        exit()

    # 逐帧读取视频并保存为图像文件
    frame_count = 0
    video_num = str(video_num).zfill(4)
    save_dir = os.path.join(output_dir, str(video_num), 'images')
    os.makedirs(save_dir, exist_ok=True)
    while True:
        # 读取一帧
        ret, frame = cap.read()

        # 如果成功读取到帧
        if ret:
            # 生成保存图像的文件名
            tmp = str(frame_count).zfill(4)
            image_path = os.path.join(save_dir, f"{tmp}.png")

            # 保存帧为图像文件
            if frame_count % save_interval == 0:
                cv2.imwrite(image_path, frame)

            # 增加帧计数器
            frame_count += 1
        else:
            break

    # 释放视频流和窗口
    cap.release()

def extract_all_videos(video_dir, output_dir='/mnt/petrelfs/majie/project/diffusers/my_project/identity/aux_data/wechat-group'):
    all_videos = os.listdir(video_dir)
    for i, video in enumerate(all_videos):
        video_path = os.path.join(video_dir, video)
        extract_single_video(video_path, i, output_dir=output_dir)

def crop_single_image(image_path, output_dir):
    top, bottom = 300, 0
    left, right = 0, 0
    img = Image.open(image_path)
    width, height = img.size
    cropped_img = img.crop((left, top, width - right, height - bottom))
    os.makedirs(output_dir, exist_ok=True)
    cropped_img.save(os.path.join(output_dir, os.path.basename(image_path)))

def crop_all_images(image_dir):
    output_dir = '/mnt/petrelfs/majie/project/diffusers/my_project/identity/data/crop_frame'
    for role in os.listdir(image_dir):
        role_dir = os.path.join(image_dir, role)
        for image in os.listdir(role_dir):
            image_path = os.path.join(role_dir, image)
            crop_single_image(image_path, os.path.join(output_dir, role))

def resize_single_image(image_path, output_dir, short_size=512):
    img = Image.open(image_path)
    width, height = img.size
    if width < height:
        new_width = short_size
        new_height = int(short_size * height / width)
    else:
        new_height = short_size
        new_width = int(short_size * width / height)
    resized_img = img.resize((new_width, new_height))
    os.makedirs(output_dir, exist_ok=True)
    resized_img.save(os.path.join(output_dir, os.path.basename(image_path)))

def resize_all_images(image_dir):
    output_dir = '/mnt/petrelfs/majie/project/diffusers/my_project/identity/data2/image'
    for role in os.listdir(image_dir):
        role_dir = os.path.join(image_dir, role)
        for image in os.listdir(role_dir):
            image_path = os.path.join(role_dir, image)
            resize_single_image(image_path, os.path.join(output_dir, role))

def generate_single_canny(image_path, output_dir):
    img = cv2.imread(image_path)
    edges = cv2.Canny(img, 100, 200)
    os.makedirs(output_dir, exist_ok=True)
    Image.fromarray(edges).save(os.path.join(output_dir, os.path.basename(image_path)))

def generate_all_canny(image_dir):
    output_dir = '/mnt/petrelfs/majie/project/diffusers/my_project/identity/data/canny'
    for role in os.listdir(image_dir):
        role_dir = os.path.join(image_dir, role)
        for image in os.listdir(role_dir):
            image_path = os.path.join(role_dir, image)
            generate_single_canny(image_path, os.path.join(output_dir, role))

def generate_single_pose(image_path, output_dir, inferencer=None):
    if inferencer is None:
        print('inferencer is None')
        pose2d='/mnt/petrelfs/majie/project/openmmlab/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288.py'
        pose2d_weights='https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288-70d7ab01_20220913.pth'

        # 使用模型别名创建推断器
        inferencer = MMPoseInferencer(
            pose2d=pose2d,
            pose2d_weights=pose2d_weights
        )

    result_generator = inferencer(image_path, pred_out_dir=output_dir, vis_out_dir=output_dir, kpt_thr=0.5, thickness=3, radius=5,
                                  skeleton_style='openpose', black_background=True)
    result = next(result_generator)

def generate_all_poses(image_dir, output_dir):
    # output_dir = image_dir

    pose2d='/mnt/petrelfs/majie/project/openmmlab/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288.py'
    pose2d_weights='https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288-70d7ab01_20220913.pth'

    # 使用模型别名创建推断器
    inferencer = MMPoseInferencer(
        pose2d=pose2d,
        pose2d_weights=pose2d_weights
    )

    for role in os.listdir(image_dir):
        role_dir = os.path.join(image_dir, role, 'images')

        all_images = os.listdir(role_dir)
        pose_path = os.path.join(output_dir, role, 'pose')
        if os.path.exists(pose_path):
            all_poses = os.listdir(pose_path)
            if len(all_poses) == 2 * len(all_images):
                print(f'{role} has been processed')
                continue

        for image in os.listdir(role_dir):
            image_path = os.path.join(role_dir, image)
            generate_single_pose(image_path, os.path.join(output_dir, role, 'pose'), inferencer)

def generate_tiktok_single_pose(image_path, output_dir, inferencer=None):
    if inferencer is None:
        print('inferencer is None')
        pose2d='configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288.py'
        pose2d_weights='https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288-70d7ab01_20220913.pth'

        # 使用模型别名创建推断器
        inferencer = MMPoseInferencer(
            pose2d=pose2d,
            pose2d_weights=pose2d_weights
        )

    result_generator = inferencer(image_path, pred_out_dir=output_dir, vis_out_dir=output_dir, kpt_thr=0.5, thickness=3, radius=5,
                                  skeleton_style='openpose', black_background=True)
    result = next(result_generator)

def generate_tiktok_all_poses(image_dir):
    pose2d='/mnt/petrelfs/majie/project/openmmlab/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288.py'
    pose2d_weights='https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288-70d7ab01_20220913.pth'

    # 使用模型别名创建推断器
    inferencer = MMPoseInferencer(
        pose2d=pose2d,
        pose2d_weights=pose2d_weights
    )

    for role in os.listdir(image_dir):
        print(f'current role: {role}/{len(os.listdir(image_dir))}')
        role_dir = os.path.join(image_dir, role)
        for image in os.listdir(os.path.join(role_dir, 'images')):
            output_dir = f'/mnt/petrelfs/majie/datasets/TikTok/TikTok_dataset/{role}/pose'
            image_path = os.path.join(role_dir, 'images', image)
            generate_tiktok_single_pose(image_path, os.path.join(output_dir, role), inferencer)

if __name__ == '__main__':
    exit(0)
    # path = '/mnt/petrelfs/majie/datasets/UBC_Fashion/raw_video/test'
    # output_dir = '/mnt/petrelfs/majie/datasets/UBC_Fashion/data/test'
    # extract_all_videos(path, output_dir)

    # path = '/mnt/petrelfs/majie/project/diffusers/my_project/identity/data/ori_frame'
    # crop_all_images(path)  

    # path = '/mnt/petrelfs/majie/project/diffusers/my_project/identity/data2/raw_data'
    # resize_all_images(path)    
    
    path = '/mnt/petrelfs/majie/datasets/UBC_Fashion/data/test'
    output_dir = '/mnt/petrelfs/majie/datasets/UBC_Fashion/data/test'
    generate_all_poses(path, output_dir)

    # path = '/mnt/petrelfs/majie/project/diffusers/my_project/identity/data/resize_frame'
    # generate_all_canny(path)
    
    # path = '/mnt/petrelfs/majie/datasets/TikTok/TikTok_dataset'
    # generate_tiktok_all_poses(path)

    pass