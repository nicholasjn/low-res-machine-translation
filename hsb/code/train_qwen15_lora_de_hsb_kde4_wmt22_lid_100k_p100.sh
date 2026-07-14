#!/usr/bin/env bash
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=qwen15_de_hsb_100k
#SBATCH --output=/dss/dsshome1/04/go82lec2/NLP-26SS/logs/qwen15_de_hsb_100k_%j.out
#SBATCH --error=/dss/dsshome1/04/go82lec2/NLP-26SS/logs/qwen15_de_hsb_100k_%j.err

set -euo pipefail

cd ~/NLP-26SS/CODE
mkdir -p ../logs ../models

hostname
date
nvidia-smi

python3 train_qwen_lora.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --train_src_path ../data/splits_kde4_wmt22_lid_100k/train.de \
  --train_tgt_path ../data/splits_kde4_wmt22_lid_100k/train.hsb \
  --dev_src_path ../data/splits_kde4_wmt22_lid_100k/dev.de \
  --dev_tgt_path ../data/splits_kde4_wmt22_lid_100k/dev.hsb \
  --output_dir ../models/qwen15-lora-de-hsb-kde4-wmt22-lid-100k-1epoch \
  --num_train_epochs 1 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_length 256 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --load_in_4bit True
