#!/bin/bash

#SBATCH -J thesis
#SBATCH --gres=gpu:pascal:2
#SBATCH --mail-user=schaikejgv@gmail.com
#SBATCH --output=thesis_%j.log

module load nvidia/cuda-11.3
module load nvidia/cuda-11.3_cudnn-8.2

echo "gpu devices: "$CUDA_VISIBLE_DEVICES

python main.py
