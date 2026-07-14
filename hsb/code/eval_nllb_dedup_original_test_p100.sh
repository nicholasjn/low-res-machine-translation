#!/usr/bin/env bash
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --job-name=eval_nllb_dedup
#SBATCH --output=../logs/eval_nllb_dedup_%j.out
#SBATCH --error=../logs/eval_nllb_dedup_%j.err

set -euo pipefail

cd ~/NLP-26SS/CODE
mkdir -p ../logs ../outputs

run_nllb() {
  local direction="$1"
  local split_dir="$2"
  local model_path="$3"
  local output_path="$4"

  if [[ ! -d "$model_path" ]]; then
    echo "Skipping missing model path: $model_path"
    return 0
  fi

  python3 evaluate_extended_nllb_baseline.py \
    --direction "$direction" \
    --split_dir "$split_dir" \
    --model_path "$model_path" \
    --output_path "$output_path" \
    --num_test 2000
}

run_nllb de-hsb \
  ../data/splits_original_test_dedup_vs_original_train \
  ../models/nllb-600M-de-hsb-finetuned \
  ../outputs/nllb_finetuned_de-hsb_original_test_dedup_vs_original_train_predictions.txt

run_nllb hsb-de \
  ../data/splits_original_test_dedup_vs_original_train \
  ../models/nllb-600M-hsb-de-finetuned \
  ../outputs/nllb_finetuned_hsb-de_original_test_dedup_vs_original_train_predictions.txt

run_nllb de-hsb \
  ../data/splits_original_test_dedup_vs_cleaned_train \
  ../models/nllb-600M-de-hsb-cleaned-finetuned \
  ../outputs/nllb_cleaned_finetuned_de-hsb_original_test_dedup_vs_cleaned_train_predictions.txt

run_nllb hsb-de \
  ../data/splits_original_test_dedup_vs_cleaned_train \
  ../models/nllb-600M-hsb-de-cleaned-finetuned \
  ../outputs/nllb_cleaned_finetuned_hsb-de_original_test_dedup_vs_cleaned_train_predictions.txt
