from setuptools import setup, find_packages
import os


def get_data_files(data_dir, prefix=''):
    file_dict = {}
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            if prefix+root not in file_dict:
                file_dict[prefix+root] = []
            file_dict[prefix+root].append(os.path.join(root, name))
    return [(k, v) for k, v in file_dict.items()]


setup(
    name='animate',
    packages=find_packages(),

    data_files=[
        *get_data_files('animate/config', 'animate_character'),
    ]

)