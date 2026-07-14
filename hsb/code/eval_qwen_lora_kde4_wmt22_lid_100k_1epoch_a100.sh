#!/usr/bin/env bash
#SBATCH --partition=lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=eval_qwen_100k
#SBATCH --output=/dss/dsshome1/04/go82lec2/NLP-26SS/logs/eval_qwen_100k_%j.out
#SBATCH --error=/dss/dsshome1/04/go82lec2/NLP-26SS/logs/eval_qwen_100k_%j.err

set -euo pipefail

cd ~/NLP-26SS/CODE
mkdir -p ../outputs ../logs

hostname
date
nvidia-smi

python3 evaluate_qwen_lora.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --adapter_path ../models/qwen-7b-lora-de-hsb-kde4-wmt22-lid-100k-1epoch \
  --direction de-hsb \
  --src_path ../data/splits_kde4_wmt22_lid_100k/test.de \
  --ref_path ../data/splits_kde4_wmt22_lid_100k/test.hsb \
  --output_path ../outputs/qwen7b_lora_de-hsb_kde4_wmt22_lid_100k_1epoch_predictions.txt \
  --num_test 2000 \
  --load_in_4bit True

python3 evaluate_qwen_lora.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --adapter_path ../models/qwen-7b-lora-hsb-de-kde4-wmt22-lid-100k-1epoch \
  --direction hsb-de \
  --src_path ../data/splits_kde4_wmt22_lid_100k/test.hsb \
  --ref_path ../data/splits_kde4_wmt22_lid_100k/test.de \
  --output_path ../outputs/qwen7b_lora_hsb-de_kde4_wmt22_lid_100k_1epoch_predictions.txt \
  --num_test 2000 \
  --load_in_4bit True
