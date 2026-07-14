#!/usr/bin/env bash
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=eval_pipe2_all
#SBATCH --output=/dss/dsshome1/04/go82lec2/NLP-26SS/logs/eval_pipe2_all_%j.out
#SBATCH --error=/dss/dsshome1/04/go82lec2/NLP-26SS/logs/eval_pipe2_all_%j.err

set -euo pipefail

cd ~/NLP-26SS/CODE
mkdir -p ../outputs ../outputs/xcomet ../logs

echo "===== Job info ====="
hostname
date
nvidia-smi

echo "===== BLEU/chrF++: de->hsb ====="
python3 evaluate_extended_nllb_baseline.py \
  --model_path ../models/nllb-600M-de-hsb-pipeline2-mt-de-51k-full \
  --src_path ../data/splits_kde4_pipeline2_mt_de_51k/test.de \
  --ref_path ../data/splits_kde4_pipeline2_mt_de_51k/test.hsb \
  --direction de-hsb \
  --output_path ../outputs/nllb_de-hsb_pipeline2_mt_de_51k_predictions.txt \
  --num_test 2000

echo "===== BLEU/chrF++: hsb->de ====="
python3 evaluate_extended_nllb_baseline.py \
  --model_path ../models/nllb-600M-hsb-de-pipeline2-mt-de-51k-full \
  --src_path ../data/splits_kde4_pipeline2_mt_de_51k/test.hsb \
  --ref_path ../data/splits_kde4_pipeline2_mt_de_51k/test.de \
  --direction hsb-de \
  --output_path ../outputs/nllb_hsb-de_pipeline2_mt_de_51k_predictions.txt \
  --num_test 2000

echo "===== xCOMET: de->hsb ====="
python3 evaluate_xcomet.py \
  --src ../data/splits_kde4_pipeline2_mt_de_51k/test.de \
  --mt ../outputs/nllb_de-hsb_pipeline2_mt_de_51k_predictions.txt \
  --ref ../data/splits_kde4_pipeline2_mt_de_51k/test.hsb \
  --output_json ../outputs/xcomet/nllb_de-hsb_pipeline2_mt_de_51k_xcomet.json \
  --output_tsv ../outputs/xcomet/nllb_de-hsb_pipeline2_mt_de_51k_xcomet.tsv

echo "===== xCOMET: hsb->de ====="
python3 evaluate_xcomet.py \
  --src ../data/splits_kde4_pipeline2_mt_de_51k/test.hsb \
  --mt ../outputs/nllb_hsb-de_pipeline2_mt_de_51k_predictions.txt \
  --ref ../data/splits_kde4_pipeline2_mt_de_51k/test.de \
  --output_json ../outputs/xcomet/nllb_hsb-de_pipeline2_mt_de_51k_xcomet.json \
  --output_tsv ../outputs/xcomet/nllb_hsb-de_pipeline2_mt_de_51k_xcomet.tsv

echo "===== Done ====="
date
