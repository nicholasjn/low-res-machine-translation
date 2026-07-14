#!/usr/bin/env bash
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=nllb_wmt22_100k
#SBATCH --output=/dss/dsshome1/04/go821ec2/NLP-26SS/logs/nllb_wmt22_100k_%j.out
#SBATCH --error=/dss/dsshome1/04/go821ec2/NLP-26SS/logs/nllb_wmt22_100k_%j.err

set -euo pipefail

cd ~/NLP-26SS/CODE
mkdir -p ../logs ../models

hostname
nvidia-smi

python3 train_nllb_finetune.py \
  --direction de-hsb \
  --split_dir ../data/splits_kde4_wmt22_lid_100k \
  --num_train_epochs 5 \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_length 128 \
  --output_dir ../models/nllb-600M-de-hsb-kde4-wmt22-lid-100k

python3 train_nllb_finetune.py \
  --direction hsb-de \
  --split_dir ../data/splits_kde4_wmt22_lid_100k \
  --num_train_epochs 5 \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_length 128 \
  --output_dir ../models/nllb-600M-hsb-de-kde4-wmt22-lid-100k
