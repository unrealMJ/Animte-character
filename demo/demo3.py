import json
from tqdm import tqdm
import os

def remove_not_exist(data):
    data_root = '/mnt/petrelfs/majie/project/HumanSD/humansd_data/datasets/Laion'
    keys = list(data.keys())
    values = list(data.values())
    filter_data = []
    for i in tqdm(range(len(values))):
        each = values[i]
        each['key'] = keys[i]
        each['img_path'] = os.path.join(data_root, each['img_path'])
        if os.path.exists(each['img_path']):
            filter_data.append(each)
    return filter_data


json_file = '/mnt/petrelfs/majie/project/HumanSD/humansd_data/datasets/Laion/Aesthetics_Human/mapping_file_training.json'

with open(json_file) as f:
    data = json.load(f)

print(f'Dataset size: {len(data)}')
data = remove_not_exist(data)

new_dict = {}
for each in data:
    key = each['key']
    each.pop('key')
    new_dict[key] = each

print(f'Filter dataset size: {len(new_dict)}')

# write
save_path = f'/mnt/petrelfs/majie/project/My-IP-Adapter/data/mapping_file_training.json'
with open(save_path, 'w') as f:
    json.dump(new_dict, f)