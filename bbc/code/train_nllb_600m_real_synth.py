import os
import re
import torch
import pandas as pd
import numpy as np

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from sacrebleu.metrics import BLEU, CHRF

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Run this script inside a GPU SLURM job.")

# ============================================================
# Config
# ============================================================

PROJECT_DIR = os.environ.get("PROJECT_DIR", "/dss/dsshome1/0C/go93pec2/batak-nllb")

DATA_PATH = f"{PROJECT_DIR}/data/synthetic_66k_plus_nusax_train.jsonl"
VALIDATION_PATH = f"{PROJECT_DIR}/data/valid_real.jsonl"

OUTPUT_DIR = f"{PROJECT_DIR}/outputs/nllb-600m-batak-5epoch-real-synt"

MODEL_NAME = "facebook/nllb-200-distilled-600M"

SEED = 42
MAX_LENGTH = 256

TGT_LANG = "bbc_Latn"
INIT_LANG = "ind_Latn"

LANG_MAP = {
    "id": "ind_Latn",
    "en": "eng_Latn",
    "eng": "eng_Latn",
    "bbc": "bbc_Latn",
    "ind": "ind_Latn"
}

# ============================================================
# Load dataset
# ============================================================

print("Loading train datasets from:", DATA_PATH)
with open(DATA_PATH, "r", encoding="utf-8") as f:
    train_df = pd.read_json(f, lines=True)

print("Loading validation dataset from:", VALIDATION_PATH)
with open(VALIDATION_PATH, "r", encoding="utf-8") as f:
    val_df = pd.read_json(f, lines=True)

required_cols = {"src_lang", "tgt_lang", "source", "target"}
if missing_train := required_cols - set(train_df.columns):
    raise ValueError(f"Missing columns in training JSONL: {missing_train}")
if missing_val := required_cols - set(val_df.columns):
    raise ValueError(f"Missing columns in validation JSONL: {missing_val}")

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df, preserve_index=False),
    "validation": Dataset.from_pandas(val_df, preserve_index=False),
})

# ============================================================
# Clean and filter
# ============================================================

def clean_example(ex):
    ex["source"] = re.sub(r"\s+", " ", str(ex["source"]).strip())
    ex["target"] = re.sub(r"\s+", " ", str(ex["target"]).strip())
    return ex

def valid_example(ex):
    # Ensure source and target are not empty
    if len(ex["source"]) == 0 or len(ex["target"]) == 0:
        return False
    # Strictly enforce that the target language is Batak
    if LANG_MAP.get(ex["tgt_lang"], ex["tgt_lang"]) != TGT_LANG:
        return False
    return True

def length_filter(ex, max_chars=1000, ratio_limit=3.0):
    src, tgt = ex["source"], ex["target"]
    if len(src) > max_chars or len(tgt) > max_chars:
        return False
    src_len, tgt_len = max(len(src.split()), 1), max(len(tgt.split()), 1)
    ratio = max(src_len / tgt_len, tgt_len / src_len)
    return ratio <= ratio_limit

dataset = dataset.map(clean_example).filter(valid_example).filter(length_filter)

def deduplicate_hf_dataset(ds):
    seen = set()
    keep_indices = []
    for i, ex in enumerate(ds):
        pair = (ex["source"], ex["target"])
        if pair not in seen:
            seen.add(pair)
            keep_indices.append(i)
    return ds.select(keep_indices)

dataset["train"] = deduplicate_hf_dataset(dataset["train"])
dataset["validation"] = deduplicate_hf_dataset(dataset["validation"])

print("=" * 80)
print("Clean train examples (Into Batak only):", len(dataset["train"]))
print("Clean validation examples (Into Batak only):", len(dataset["validation"]))

# ============================================================
# Load model and tokenizer
# ============================================================

print("Loading tokenizer and model:", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Add new Batak Toba language tag safely
if TGT_LANG not in tokenizer.get_vocab():
    print(f"Adding new language token: {TGT_LANG}")
    tokenizer.add_special_tokens({"additional_special_tokens": [TGT_LANG]})
    model.resize_token_embeddings(len(tokenizer))

    new_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    init_id = tokenizer.convert_tokens_to_ids(INIT_LANG)

    print("New Batak token id:", new_id, "| Init Indonesian token id:", init_id)

    # Safely initialize BOTH input embeddings and LM head
    with torch.no_grad():
        input_embeds = model.get_input_embeddings().weight
        output_embeds = model.get_output_embeddings().weight
        
        input_embeds[new_id] = input_embeds[init_id].clone()
        output_embeds[new_id] = output_embeds[init_id].clone()
else:
    print(f"{TGT_LANG} already exists in tokenizer vocab.")

model.config.use_cache = False

# Because we ONLY translate into Batak, we CAN use forced_bos_token_id globally
tgt_lang_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
model.config.forced_bos_token_id = tgt_lang_id
model.generation_config.forced_bos_token_id = tgt_lang_id

# ============================================================
# Tokenization
# ============================================================

def preprocess_function(examples):
    input_ids, attention_masks, labels = [], [], []

    for src_lang, source, target in zip(
        examples["src_lang"], examples["source"], examples["target"]
    ):
        # Source language can be ind or eng, so we set it dynamically
        tokenizer.src_lang = LANG_MAP.get(src_lang, src_lang)
        tokenizer.tgt_lang = TGT_LANG

        # text_target automatically applies the correct tgt_lang token as BOS
        encoded = tokenizer(
            source,
            text_target=target,
            max_length=MAX_LENGTH,
            truncation=True,
        )

        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])
        labels.append(encoded["labels"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# ============================================================
# Evaluation Metric 
# ============================================================

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.asarray(preds)
    labels = np.asarray(labels)

    pad_id = tokenizer.pad_token_id
    vocab_size = len(tokenizer)

    labels = np.where(labels == -100, pad_id, labels)
    preds = np.where(preds == -100, pad_id, preds)
    preds = np.where(preds < 0, pad_id, preds)
    labels = np.where(labels < 0, pad_id, labels)
    preds = np.where(preds >= vocab_size, pad_id, preds)
    labels = np.where(labels >= vocab_size, pad_id, labels)

    decoded_preds = [pred.strip() for pred in tokenizer.batch_decode(preds.astype(np.int64), skip_special_tokens=True)]
    decoded_labels = [label.strip() for label in tokenizer.batch_decode(labels.astype(np.int64), skip_special_tokens=True)]

    chrf = CHRF(word_order=2)
    bleu = BLEU(tokenize="flores200")

    return {
        "chrF++": round(chrf.corpus_score(decoded_preds, [decoded_labels]).score, 2),
        "BLEU": round(bleu.corpus_score(decoded_preds, [decoded_labels]).score, 2),
    }

# ============================================================
# Training arguments
# ============================================================

bf16_available = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=3e-5,
    warmup_ratio=0.10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    predict_with_generate=True,
    generation_max_length=256,
    generation_num_beams=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="chrF++",
    greater_is_better=True,
    fp16=not bf16_available,
    bf16=bf16_available,
    gradient_checkpointing=True,
    optim="adamw_torch",
    report_to="none",
    dataloader_num_workers=2,
    remove_unused_columns=True,
)

# ============================================================
# Trainer
# ============================================================

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ============================================================
# Train and save
# ============================================================

print("Starting training...")
trainer.train()

print("Saving model to:", OUTPUT_DIR)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done.")