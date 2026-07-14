#!/usr/bin/env bash
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=xcomet_qwen_100k
#SBATCH --output=/dss/dsshome1/04/go82lec2/NLP-26SS/logs/xcomet_qwen_100k_%j.out
#SBATCH --error=/dss/dsshome1/04/go82lec2/NLP-26SS/logs/xcomet_qwen_100k_%j.err

set -euo pipefail

cd ~/NLP-26SS/CODE
mkdir -p ../outputs/xcomet ../logs

hostname
date
nvidia-smi

python3 evaluate_xcomet.py \
  --src ../data/splits_kde4_wmt22_lid_100k/test.de \
  --mt ../outputs/qwen7b_lora_de-hsb_kde4_wmt22_lid_100k_1epoch_predictions.txt \
  --ref ../data/splits_kde4_wmt22_lid_100k/test.hsb \
  --output_json ../outputs/xcomet/qwen7b_lora_de-hsb_kde4_wmt22_lid_100k_1epoch_xcomet.json \
  --output_tsv ../outputs/xcomet/qwen7b_lora_de-hsb_kde4_wmt22_lid_100k_1epoch_xcomet.tsv

python3 evaluate_xcomet.py \
  --src ../data/splits_kde4_wmt22_lid_100k/test.hsb \
  --mt ../outputs/qwen7b_lora_hsb-de_kde4_wmt22_lid_100k_1epoch_predictions.txt \
  --ref ../data/splits_kde4_wmt22_lid_100k/test.de \
  --output_json ../outputs/xcomet/qwen7b_lora_hsb-de_kde4_wmt22_lid_100k_1epoch_xcomet.json \
  --output_tsv ../outputs/xcomet/qwen7b_lora_hsb-de_kde4_wmt22_lid_100k_1epoch_xcomet.tsv
