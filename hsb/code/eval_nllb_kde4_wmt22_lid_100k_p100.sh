#!/usr/bin/env bash
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --job-name=eval_nllb_100k
#SBATCH --output=/dss/dsshome1/04/go82lec2/NLP-26SS/logs/eval_nllb_100k_%j.out
#SBATCH --error=/dss/dsshome1/04/go82lec2/NLP-26SS/logs/eval_nllb_100k_%j.err

set -euo pipefail

cd ~/NLP-26SS/CODE
mkdir -p ../logs ../outputs

hostname
nvidia-smi
date

python3 evaluate_extended_nllb_baseline.py \
  --direction de-hsb \
  --split_dir ../data/splits_kde4_wmt22_lid_100k \
  --model_path ../models/nllb-600M-de-hsb-kde4-wmt22-lid-100k \
  --output_path ../outputs/nllb_de-hsb_kde4_wmt22_lid_100k_predictions.txt \
  --num_test 2000

python3 evaluate_extended_nllb_baseline.py \
  --direction hsb-de \
  --split_dir ../data/splits_kde4_wmt22_lid_100k \
  --model_path ../models/nllb-600M-hsb-de-kde4-wmt22-lid-100k \
  --output_path ../outputs/nllb_hsb-de_kde4_wmt22_lid_100k_predictions.txt \
  --num_test 2000
