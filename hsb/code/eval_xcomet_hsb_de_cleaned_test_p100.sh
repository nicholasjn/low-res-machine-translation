#!/usr/bin/env bash
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=xcomet_hsb_de_cleaned
#SBATCH --output=../logs/xcomet_hsb_de_cleaned_%j.out
#SBATCH --error=../logs/xcomet_hsb_de_cleaned_%j.err

set -euo pipefail

cd ~/NLP-26SS/CODE
mkdir -p ../logs ../outputs/xcomet

python3 -m pip install --user "unbabel-comet>=2.2.0"

SRC=../data/splits_cleaned/test.hsb
REF=../data/splits_cleaned/test.de
MT=../outputs/nllb_cleaned_finetuned_hsb-de_cleaned_test_predictions.txt
PREFIX=../outputs/xcomet/nllb_cleaned_finetuned_hsb-de_cleaned_test

if [[ ! -f "$MT" ]]; then
  echo "Skipping missing file: $MT"
  exit 0
fi

python3 evaluate_xcomet.py \
  --src "$SRC" \
  --mt "$MT" \
  --ref "$REF" \
  --output_json "${PREFIX}_xcomet.json" \
  --output_tsv "${PREFIX}_xcomet.tsv" \
  --batch_size 1 \
  --gpus 1
