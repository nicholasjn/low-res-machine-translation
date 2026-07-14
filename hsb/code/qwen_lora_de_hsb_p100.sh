#!/usr/bin/env bash
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=qwen7b_lora_de_hsb_p100
#SBATCH --output=../logs/qwen7b_lora_de_hsb_p100_%j.out
#SBATCH --error=../logs/qwen7b_lora_de_hsb_p100_%j.err

set -euo pipefail

cd ~/NLP-26SS/CODE
mkdir -p ../logs ../models

hostname
nvidia-smi

python3 train_qwen_lora.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --train_src_path ../data/splits/train.de \
  --train_tgt_path ../data/splits/train.hsb \
  --dev_src_path ../data/splits/dev.de \
  --dev_tgt_path ../data/splits/dev.hsb \
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
  --output_dir ../models/qwen-7b-lora-de-hsb
