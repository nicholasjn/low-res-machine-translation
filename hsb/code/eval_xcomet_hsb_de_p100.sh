#!/usr/bin/env bash
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=xcomet_hsb_de
#SBATCH --output=../logs/xcomet_hsb_de_%j.out
#SBATCH --error=../logs/xcomet_hsb_de_%j.err

set -euo pipefail

cd ~/NLP-26SS/CODE
mkdir -p ../logs ../outputs/xcomet

python3 -m pip install --user "unbabel-comet>=2.2.0"

SRC=../data/splits/test.hsb
REF=../data/splits/test.de

run_xcomet() {
  local mt_file="$1"
  local prefix="$2"

  if [[ ! -f "$mt_file" ]]; then
    echo "Skipping missing file: $mt_file"
    return 0
  fi

  python3 evaluate_xcomet.py \
    --src "$SRC" \
    --mt "$mt_file" \
    --ref "$REF" \
    --output_json "${prefix}_xcomet.json" \
    --output_tsv "${prefix}_xcomet.tsv" \
    --batch_size 1 \
    --gpus 1
}

run_xcomet ../outputs/extended_nllb_hsb-de_predictions.txt \
  ../outputs/xcomet/extended_nllb_hsb-de_original_test

run_xcomet ../outputs/nllb_finetuned_hsb-de_original_test_predictions.txt \
  ../outputs/xcomet/nllb_finetuned_hsb-de_original_test

run_xcomet ../outputs/nllb_cleaned_finetuned_hsb-de_original_test_predictions.txt \
  ../outputs/xcomet/nllb_cleaned_finetuned_hsb-de_original_test

run_xcomet ../outputs/qwen_hsb-de_predictions.txt \
  ../outputs/xcomet/qwen7b_baseline_hsb-de_original_test

run_xcomet ../outputs/qwen7b_lora_hsb-de_predictions.txt \
  ../outputs/xcomet/qwen7b_lora_hsb-de_original_test
