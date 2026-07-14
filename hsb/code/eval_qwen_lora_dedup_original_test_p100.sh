#!/usr/bin/env bash
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=eval_qwen_lora_dedup
#SBATCH --output=../logs/eval_qwen_lora_dedup_%j.out
#SBATCH --error=../logs/eval_qwen_lora_dedup_%j.err

set -euo pipefail

cd ~/NLP-26SS/CODE
mkdir -p ../logs ../outputs

if [[ ! -f evaluate_qwen_lora.py ]]; then
  echo "Skipping Qwen LoRA dedup evaluation: CODE/evaluate_qwen_lora.py was not found."
  exit 0
fi

run_qwen_lora() {
  local direction="$1"
  local src="$2"
  local ref="$3"
  local adapter="$4"
  local output="$5"

  if [[ ! -d "$adapter" ]]; then
    echo "Skipping missing adapter path: $adapter"
    return 0
  fi

  python3 evaluate_qwen_lora.py \
    --direction "$direction" \
    --src_path "$src" \
    --ref_path "$ref" \
    --adapter_path "$adapter" \
    --output_path "$output" \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --load_in_4bit True \
    --num_test 2000
}

run_qwen_lora de-hsb \
  ../data/splits_original_test_dedup_vs_original_train/test.de \
  ../data/splits_original_test_dedup_vs_original_train/test.hsb \
  ../models/qwen-7b-lora-de-hsb \
  ../outputs/qwen7b_lora_de-hsb_original_test_dedup_vs_original_train_predictions.txt

run_qwen_lora hsb-de \
  ../data/splits_original_test_dedup_vs_original_train/test.hsb \
  ../data/splits_original_test_dedup_vs_original_train/test.de \
  ../models/qwen-7b-lora-hsb-de \
  ../outputs/qwen7b_lora_hsb-de_original_test_dedup_vs_original_train_predictions.txt
