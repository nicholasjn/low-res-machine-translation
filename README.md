# Fine-tuning NLLB-200-600M for Batak Translation

This repository/notebook fine-tunes `facebook/nllb-200-distilled-600M` for a low-resource translation direction such as:

- Indonesian → Batak Toba
- English → Batak Toba

The workflow follows the constrained-submission idea from the WMT24 low-resource MT paper: clean the available parallel data, add a new language tag for an unsupported language, fine-tune NLLB-200-600M, evaluate with BLEU/chrF++, and save the best checkpoint.

Install dependencies:

```python
!pip install -q transformers datasets accelerate sentencepiece sacrebleu evaluate
```

Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```
