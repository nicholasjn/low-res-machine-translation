#!/usr/bin/env bash
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --job-name=eval_qwen15_100k
#SBATCH --output=/dss/dsshome1/04/go82lec2/NLP-26SS/logs/eval_qwen15_100k_%j.out
#SBATCH --error=/dss/dsshome1/04/go82lec2/NLP-26SS/logs/eval_qwen15_100k_%j.err

set -euo pipefail

cd ~/NLP-26SS/CODE
mkdir -p ../outputs ../outputs/xcomet ../logs

echo "===== Job info ====="
hostname
date
nvidia-smi

echo "===== BLEU/chrF++: de->hsb ====="
python3 evaluate_qwen_lora.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path ../models/qwen15-lora-de-hsb-kde4-wmt22-lid-100k-1epoch \
  --src_path ../data/splits_kde4_wmt22_lid_100k/test.de \
  --ref_path ../data/splits_kde4_wmt22_lid_100k/test.hsb \
  --output_path ../outputs/qwen15_lora_de-hsb_kde4_wmt22_lid_100k_1epoch_predictions.txt \
  --num_test 2000 \
  --max_new_tokens 128 \
  --load_in_4bit True

echo "===== BLEU/chrF++: hsb->de ====="
python3 evaluate_qwen_lora.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path ../models/qwen15-lora-hsb-de-kde4-wmt22-lid-100k-1epoch \
  --src_path ../data/splits_kde4_wmt22_lid_100k/test.hsb \
  --ref_path ../data/splits_kde4_wmt22_lid_100k/test.de \
  --output_path ../outputs/qwen15_lora_hsb-de_kde4_wmt22_lid_100k_1epoch_predictions.txt \
  --num_test 2000 \
  --max_new_tokens 128 \
  --load_in_4bit True

echo "===== xCOMET: de->hsb ====="
python3 evaluate_xcomet.py \
  --src ../data/splits_kde4_wmt22_lid_100k/test.de \
  --mt ../outputs/qwen15_lora_de-hsb_kde4_wmt22_lid_100k_1epoch_predictions.txt \
  --ref ../data/splits_kde4_wmt22_lid_100k/test.hsb \
  --output_json ../outputs/xcomet/qwen15_lora_de-hsb_kde4_wmt22_lid_100k_1epoch_xcomet.json \
  --output_tsv ../outputs/xcomet/qwen15_lora_de-hsb_kde4_wmt22_lid_100k_1epoch_xcomet.tsv

echo "===== xCOMET: hsb->de ====="
python3 evaluate_xcomet.py \
  --src ../data/splits_kde4_wmt22_lid_100k/test.hsb \
  --mt ../outputs/qwen15_lora_hsb-de_kde4_wmt22_lid_100k_1epoch_predictions.txt \
  --ref ../data/splits_kde4_wmt22_lid_100k/test.de \
  --output_json ../outputs/xcomet/qwen15_lora_hsb-de_kde4_wmt22_lid_100k_1epoch_xcomet.json \
  --output_tsv ../outputs/xcomet/qwen15_lora_hsb-de_kde4_wmt22_lid_100k_1epoch_xcomet.tsv

echo "===== Done ====="
date
