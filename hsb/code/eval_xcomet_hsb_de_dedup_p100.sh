#!/usr/bin/env bash
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=xcomet_hsb_de_dedup
#SBATCH --output=../logs/xcomet_hsb_de_dedup_%j.out
#SBATCH --error=../logs/xcomet_hsb_de_dedup_%j.err

set -euo pipefail

cd ~/NLP-26SS/CODE
mkdir -p ../logs ../outputs/xcomet

python3 -m pip install --user "unbabel-comet>=2.2.0"

run_xcomet() {
  local src="$1"
  local ref="$2"
  local mt="$3"
  local prefix="$4"

  if [[ ! -f "$mt" ]]; then
    echo "Skipping missing MT file: $mt"
    return 0
  fi

  python3 evaluate_xcomet.py \
    --src "$src" \
    --mt "$mt" \
    --ref "$ref" \
    --output_json "${prefix}_xcomet.json" \
    --output_tsv "${prefix}_xcomet.tsv" \
    --batch_size 1 \
    --gpus 1
}

run_xcomet \
  ../data/splits_original_test_dedup_vs_original_train/test.hsb \
  ../data/splits_original_test_dedup_vs_original_train/test.de \
  ../outputs/nllb_finetuned_hsb-de_original_test_dedup_vs_original_train_predictions.txt \
  ../outputs/xcomet/nllb_finetuned_hsb-de_original_test_dedup_vs_original_train

run_xcomet \
  ../data/splits_original_test_dedup_vs_cleaned_train/test.hsb \
  ../data/splits_original_test_dedup_vs_cleaned_train/test.de \
  ../outputs/nllb_cleaned_finetuned_hsb-de_original_test_dedup_vs_cleaned_train_predictions.txt \
  ../outputs/xcomet/nllb_cleaned_finetuned_hsb-de_original_test_dedup_vs_cleaned_train

run_xcomet \
  ../data/splits_original_test_dedup_vs_original_train/test.hsb \
  ../data/splits_original_test_dedup_vs_original_train/test.de \
  ../outputs/qwen7b_lora_hsb-de_original_test_dedup_vs_original_train_predictions.txt \
  ../outputs/xcomet/qwen7b_lora_hsb-de_original_test_dedup_vs_original_train
