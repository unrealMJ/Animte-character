import os
import shutil
from tqdm import tqdm
import concurrent.futures
from scenedetect import detect, ContentDetector, split_video_ffmpeg, ThresholdDetector, AdaptiveDetector



def count_num(path):
    pbar.update()
    scene_list = detect(path, AdaptiveDetector())
    root = '/mnt/petrelfs/share_data/majie/douyin'
    if len(scene_list) == 0:
        dir_name = path.split('/')[-2]
        file_name = path.split('/')[-1]
        dst_dir = os.path.join(root, dir_name)
        os.makedirs(dst_dir, exist_ok=True)
        # print(os.path.join(dst_dir, file_name))
        shutil.copy(path, os.path.join(dst_dir, file_name))
        return 1
    else:
        return 0


if __name__ == '__main__':
    all_files = []
    path = '/mnt/petrelfs/majie/datasets/douyin/raw_video'

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.mp4'):
                all_files.append(os.path.join(root, file))
    
    pbar = tqdm(total=len(all_files))
    update = lambda *args: pbar.update()

    # with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
    #     pool_outputs = executor.map(count_num, all_files)
    #     # pool_outputs = executor.map_async(count_num, all_files, callback=update)
    #     # executor.join()
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        pool_outputs = executor.map(count_num, all_files)
        # executor.join()
    pool_outputs = list(pool_outputs)
    print(sum(pool_outputs))



