import os
import shutil

# root = 'output'

for root, dirs, files in os.walk('output'):
    for name in files:
        if name == 'optimizer.bin':
            # delete
            print(os.path.join(root, name))
            os.remove(os.path.join(root, name))

