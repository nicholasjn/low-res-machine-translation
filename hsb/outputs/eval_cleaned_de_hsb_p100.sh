#!/bin/bash
#SBATCH --partition=lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --job-name=eval_clean_de
#SBATCH --output=eval_clean_de_%j.out

cd ~/NLP-26SS/CODE

hostname
nvidia-smi
python3 -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

python3 -m pip install --user colorama

python3 evaluate_extended_nllb_baseline.py \
  --direction de-hsb \
  --split_dir ../data/splits \
  --model_path ../models/nllb-600M-de-hsb-cleaned-finetuned \
  --output_path ../outputs/nllb_cleaned_finetuned_de-hsb_original_test_predictions.txt \
  --num_test 2000
