#!/usr/bin/env sh

srun --partition=mm_lol \
     --quotatype=reserved \
     --job-name=ip \
     --gres=gpu:1 \
     --ntasks=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     --kill-on-bad-exit=1 \
     sh train.sh