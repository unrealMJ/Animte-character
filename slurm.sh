srun -p mm_lol \
    --quotatype=reserved \
    --job-name=train \
    --gres=gpu:1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --kill-on-bad-exit=1 \
    --pty bash


# srun -p mm_lol --job-name=identity --gres=gpu:8 --cpus-per-task=64 --pty bash

# 进入指定节点查看gpustat
# srun -p mm_lol -w SH-IDC1-10-140-24-122 --pty bash