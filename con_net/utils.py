import os
import shutil
from PIL import Image


def copy_src(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for each in os.listdir(os.getcwd()):
        if each.endswith('.py') or each.endswith('.sh'):
            shutil.copy(each, os.path.join(output_dir, each))
    shutil.copytree('con_net/', os.path.join(output_dir, 'con_net'))
    shutil.copytree('ip_adapter/', os.path.join(output_dir, 'ip_adapter'))
    shutil.copytree('config/', os.path.join(output_dir, 'config'))
    # gpt生成，判断是否正确
    # 遍历目标目录下的所有文件，并设置文件权限为只读
    # for root, dirs, files in os.walk(output_dir):
    #     for file in files:
    #         file_path = os.path.join(root, file)
    #         os.chmod(file_path, 0o444)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid