#!/usr/bin/env bash
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --time=04:00:00
#SBATCH --job-name=wmt22_lid_p100
#SBATCH --output=logs/wmt22_lid_p100_%j.out
#SBATCH --error=logs/wmt22_lid_p100_%j.err

set -euo pipefail

cd ~/NLP-26SS
mkdir -p logs outputs/lid_reports

hostname
date
python3 --version

python3 CODE/check_wmt22_lid_quality.py \
  --de_path data/wmt22_parallel/wmt22_all_rebuilt.de \
  --hsb_path data/wmt22_parallel/wmt22_all_rebuilt.hsb \
  --output_tsv outputs/lid_reports/wmt22_rebuilt_lid_full.tsv \
  --bad_tsv outputs/lid_reports/wmt22_rebuilt_lid_bad.tsv

python3 CODE/summarize_lid_report.py \
  --lid_tsv outputs/lid_reports/wmt22_rebuilt_lid_full.tsv \
  --summary_tsv outputs/lid_reports/wmt22_rebuilt_lid_summary.tsv \
  --categorized_tsv outputs/lid_reports/wmt22_rebuilt_lid_categorized.tsv
