#!/usr/bin/env bash
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=xcomet_nllb_100k
#SBATCH --output=/dss/dsshome1/04/go82lec2/NLP-26SS/logs/xcomet_nllb_100k_%j.out
#SBATCH --error=/dss/dsshome1/04/go82lec2/NLP-26SS/logs/xcomet_nllb_100k_%j.err

set -euo pipefail

cd ~/NLP-26SS/CODE
mkdir -p ../logs ../outputs/xcomet

hostname
date
nvidia-smi

python3 -m pip install --user "unbabel-comet>=2.2.0"

run_xcomet() {
  local src_file="$1"
  local ref_file="$2"
  local mt_file="$3"
  local prefix="$4"

  if [[ ! -f "$mt_file" ]]; then
    echo "Skipping missing MT file: $mt_file"
    return 0
  fi

  python3 evaluate_xcomet.py \
    --src "$src_file" \
    --mt "$mt_file" \
    --ref "$ref_file" \
    --output_json "${prefix}_xcomet.json" \
    --output_tsv "${prefix}_xcomet.tsv" \
    --batch_size 1 \
    --gpus 1
}

run_xcomet \
  ../data/splits_kde4_wmt22_lid_100k/test.de \
  ../data/splits_kde4_wmt22_lid_100k/test.hsb \
  ../outputs/nllb_de-hsb_kde4_wmt22_lid_100k_predictions.txt \
  ../outputs/xcomet/nllb_de-hsb_kde4_wmt22_lid_100k

run_xcomet \
  ../data/splits_kde4_wmt22_lid_100k/test.hsb \
  ../data/splits_kde4_wmt22_lid_100k/test.de \
  ../outputs/nllb_hsb-de_kde4_wmt22_lid_100k_predictions.txt \
  ../outputs/xcomet/nllb_hsb-de_kde4_wmt22_lid_100k
