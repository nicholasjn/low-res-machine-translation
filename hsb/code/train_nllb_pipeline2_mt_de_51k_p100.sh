#!/usr/bin/env bash
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=nllb_pipe2_full
#SBATCH --output=/dss/dsshome1/04/go82lec2/NLP-26SS/logs/nllb_pipe2_full_%j.out
#SBATCH --error=/dss/dsshome1/04/go82lec2/NLP-26SS/logs/nllb_pipe2_full_%j.err

set -euo pipefail

cd ~/NLP-26SS/CODE
mkdir -p ../logs ../models

hostname
date
nvidia-smi

python3 train_nllb_finetune.py \
  --direction de-hsb \
  --model_path ../models/nllb-600M-hsb-init \
  --split_dir ../data/splits_kde4_pipeline2_mt_de_51k \
  --train_src_path ../data/splits_kde4_pipeline2_mt_de_51k/train.de \
  --train_tgt_path ../data/splits_kde4_pipeline2_mt_de_51k/train.hsb \
  --dev_src_path ../data/splits_kde4_pipeline2_mt_de_51k/dev.de \
  --dev_tgt_path ../data/splits_kde4_pipeline2_mt_de_51k/dev.hsb \
  --num_train_epochs 5 \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_length 128 \
  --output_dir ../models/nllb-600M-de-hsb-pipeline2-mt-de-51k-full

python3 train_nllb_finetune.py \
  --direction hsb-de \
  --model_path ../models/nllb-600M-hsb-init \
  --split_dir ../data/splits_kde4_pipeline2_mt_de_51k \
  --train_src_path ../data/splits_kde4_pipeline2_mt_de_51k/train.hsb \
  --train_tgt_path ../data/splits_kde4_pipeline2_mt_de_51k/train.de \
  --dev_src_path ../data/splits_kde4_pipeline2_mt_de_51k/dev.hsb \
  --dev_tgt_path ../data/splits_kde4_pipeline2_mt_de_51k/dev.de \
  --num_train_epochs 5 \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_length 128 \
  --output_dir ../models/nllb-600M-hsb-de-pipeline2-mt-de-51k-full
