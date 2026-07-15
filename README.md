# Low-Resource Machine Translation

Translating Azerbaijani, Batak Toba, and Upper Sorbian by fine-tuning NLLB and Qwen.

This repository contains the research overview, methodology, and findings for our project exploring machine translation in low-resource environments. This research was conducted at the **Technical University of Munich (TUM)**.

## 👥 Authors
* Rauf Ibishov
* Ziyu Zhou
* Nicholas Nainggolan

---

## 📖 Overview

Thousands of low-resource languages lack usable translation tools due to a severe shortage of parallel training data. Without targeted training, even state-of-the-art models struggle—for example, NLLB achieves a very low zero-shot score on English $\leftrightarrow$ Batak Toba because the language is entirely unsupported by the base model. 

Our research investigates how far **scale, data cleaning, and synthetic data** can push the boundaries of low-resource translation. We tested these approaches across three distinct data settings using **NLLB-600M** and **Qwen2.5**.

## 🌍 Supported Languages

| Language | Pair | Context |
| :--- | :--- | :--- |
| **Azerbaijani (AZ)** | EN $\leftrightarrow$ AZ | Turkic language. Large speaker base, but severely lacks high-quality parallel resources. |
| **Batak Toba (BBC)** | EN/ID $\leftrightarrow$ BBC | Austronesian language (Indonesia). Extreme low-resource with virtually no supervised training data. |
| **Upper Sorbian (HSB)**| DE $\leftrightarrow$ HSB | West Slavic minority language (Germany). Highly constrained parallel data against the dominant local language. |

---

## ⚙️ Models & Methodology

We compared standard Encoder-Decoder architectures against Decoder-only LLMs using the following setups:

*   **NLLB-600M:** Full fine-tuning for 3 to 15 epochs depending on the language.
*   **Qwen2.5 + LoRA:** 
    *   AZ & BBC: 1.5B-Instruct ($r$=16, $\alpha$=32 for BBC; $r$=8, $\alpha$=16 for AZ).
    *   HSB: 7B-Instruct ($r$=8, $\alpha$=16).
*   **Evaluation:** chrF++, (sp)BLEU, and xCOMET. 

### Datasets
*   **AZ:** MTEB, MNLI, Back-translations (Tested on FLORES devtest).
*   **BBC:** NusaX, Wiki-Batak, MADLAD-400, Synthetic (Tested on NusaX test).
*   **HSB:** WMT22/KDE4, OPUS/Pivot-MT synth (Tested on WMT HSB devtest).

---

## 📊 Key Findings

1.  **NLLB Outperforms Qwen + LoRA:** NLLB yields higher chrF++ and BLEU scores across all six translation directions compared to Qwen. However, Qwen remains competitive on semantic evaluation metrics like xCOMET specifically for Azerbaijani.
2.  **Fine-Tuning is Essential:** Fine-tuning drastically closes the performance gap, yielding the largest gains exactly where baselines fail. As a result of this work, we contributed **new language tags to NLLB for HSB and BBC**.
3.  **Data Cleaning Drives Performance:** Rule-based cleaning significantly boosts translation quality (e.g., catching and cleaning 129 leaked test pairs in the HSB dataset). These improvements are most prominent when translating from a high-resource language to a low-resource language (e.g., German $\to$ Upper Sorbian).
4.  **Synthetic Data is Asymmetric:** In general, synthetic data helps when the low-resource language is the target, but *hurts* when it is the source. However, for extreme low-resource cases like Batak Toba, using synthetic data bilaterally yields consistent improvements in both directions.
