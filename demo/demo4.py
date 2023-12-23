import os
from PIL import Image
import cv2


# root = '/mnt/petrelfs/majie/project/My-IP-Adapter/data/test_demo/cctv_nantong/sixpose'
# out_dir = '/mnt/petrelfs/majie/project/My-IP-Adapter/data/test_demo/cctv_nantong/sixpose_resize'
# os.makedirs(out_dir, exist_ok=True)

# for each in os.listdir(root):
#     path = os.path.join(root, each)
#     img = Image.open(path)

#     # resize short size to 480
#     w, h = img.size
#     if w < h:
#         new_w = 480
#         new_h = int(h * new_w / w)
#     else:
#         new_h = 480
#         new_w = int(w * new_h / h)
#     img = img.resize((new_w, new_h))

#     img.save(os.path.join(out_dir, each))

root = '/mnt/petrelfs/majie/project/My-IP-Adapter/data/test_demo/cctv_nantong/sixpose_resize'
out_dir = '/mnt/petrelfs/majie/project/My-IP-Adapter/data/test_demo/cctv_nantong/canny'
os.makedirs(out_dir, exist_ok=True)

for each in os.listdir(root):
    path = os.path.join(root, each)
    # img = Image.open(path)
    img = cv2.imread(path)
    edges = cv2.Canny(img, 100, 200)
    Image.fromarray(edges).save(os.path.join(out_dir, each))


