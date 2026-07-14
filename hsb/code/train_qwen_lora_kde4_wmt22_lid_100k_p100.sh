#!/usr/bin/env bash
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=qwen_lora_100k
#SBATCH --output=/dss/dsshome1/04/go82lec2/NLP-26SS/logs/qwen_lora_100k_%j.out
#SBATCH --error=/dss/dsshome1/04/go82lec2/NLP-26SS/logs/qwen_lora_100k_%j.err

set -euo pipefail

cd ~/NLP-26SS/CODE
mkdir -p ../logs ../models

hostname
date
nvidia-smi

python3 train_qwen_lora.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --train_src_path ../data/splits_kde4_wmt22_lid_100k/train.de \
  --train_tgt_path ../data/splits_kde4_wmt22_lid_100k/train.hsb \
  --dev_src_path ../data/splits_kde4_wmt22_lid_100k/dev.de \
  --dev_tgt_path ../data/splits_kde4_wmt22_lid_100k/dev.hsb \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --max_length 256 \
  --load_in_4bit True \
  --output_dir ../models/qwen-7b-lora-de-hsb-kde4-wmt22-lid-100k

python3 train_qwen_lora.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --train_src_path ../data/splits_kde4_wmt22_lid_100k/train.hsb \
  --train_tgt_path ../data/splits_kde4_wmt22_lid_100k/train.de \
  --dev_src_path ../data/splits_kde4_wmt22_lid_100k/dev.hsb \
  --dev_tgt_path ../data/splits_kde4_wmt22_lid_100k/dev.de \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --max_length 256 \
  --load_in_4bit True \
  --output_dir ../models/qwen-7b-lora-hsb-de-kde4-wmt22-lid-100k
