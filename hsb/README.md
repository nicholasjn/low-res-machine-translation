# German–Upper Sorbian MT Experiments

This folder contains the Upper Sorbian (HSB) part of our low-resource MT project.

## Data

We use KDE4 German–Upper Sorbian as the base dataset.

Main data settings:

1. KDE4 original / cleaned data
2. Pipeline 1: KDE4 + WMT22 true parallel data
3. Pipeline 2: OPUS KDE4 cs/en/pl–hsb → MT-German–hsb synthetic data
4. Pipeline 3: Qwen2.5-1.5B LoRA on KDE4 + WMT22-100k

## Main Results

### Pipeline 1: true parallel WMT22-100k

| Model | Direction | BLEU | chrF++ | xCOMET |
|---|---:|---:|---:|---:|
| NLLB | de→hsb | 23.67 | 46.44 | 0.7583 |
| NLLB | hsb→de | 19.40 | 43.22 | 0.8356 |
| Qwen2.5-7B LoRA | de→hsb | 19.42 | 38.88 | 0.7565 |
| Qwen2.5-7B LoRA | hsb→de | 20.30 | 39.60 | 0.8471 |

### Pipeline 2: multilingual synthetic data

| Model | Direction | BLEU | chrF++ | xCOMET |
|---|---:|---:|---:|---:|
| NLLB | de→hsb | 40.49 | 58.65 | 0.7826 |
| NLLB | hsb→de | 6.92 | 33.19 | 0.8038 |

### Pipeline 3: Qwen2.5-1.5B LoRA

| Model | Direction | BLEU | chrF++ | xCOMET |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B LoRA | de→hsb | 16.07 | 34.59 | 0.7443 |
| Qwen2.5-1.5B LoRA | hsb→de | 16.32 | 35.36 | 0.8419 |

## Main Findings

- True parallel WMT22 data improves NLLB steadily.
- Multilingual synthetic data strongly improves de→hsb because it adds real Upper Sorbian target-side data.
- Synthetic data hurts hsb→de when the German target side is machine-translated.
- Qwen2.5-1.5B is weaker than Qwen2.5-7B on BLEU/chrF++, but xCOMET remains close.
