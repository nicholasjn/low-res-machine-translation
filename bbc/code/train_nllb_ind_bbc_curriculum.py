import os
import re
import torch
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

from datasets import Dataset
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

DATA_DIR = Path(PROJECT_DIR) / "data"

STAGE1_PATH = DATA_DIR / "train_stage1_synthetic.jsonl"
STAGE2_PATH = DATA_DIR / "train_stage2_mixed.jsonl"
STAGE3_PATH = DATA_DIR / "train_stage3_real_only.jsonl"

VALIDATION_PATH = DATA_DIR / "valid_real.jsonl"
TEST_PATH = DATA_DIR / "test_real.jsonl"

OUTPUT_DIR = Path(PROJECT_DIR) / "outputs" / "nllb-600m-batak-ind-curriculum"

MODEL_NAME = "facebook/nllb-200-distilled-600M"

SEED = 42
MAX_LENGTH = 256

# NLLB language tokens
BBC_LANG = "bbc_Latn"
INIT_LANG = "ind_Latn"

LANG_MAP = {
    "id": "ind_Latn",
    "ind": "ind_Latn",
    "ind_Latn": "ind_Latn",

    "en": "eng_Latn",
    "eng": "eng_Latn",
    "eng_Latn": "eng_Latn",

    "bbc": "bbc_Latn",
    "bbc_Latn": "bbc_Latn",
}

STAGES = [
    {
        "name": "stage1_synthetic",
        "train_path": STAGE1_PATH,
        "learning_rate": 5e-5,
        "num_train_epochs": 5,
        "warmup_ratio": 0.10,
    },
    {
        "name": "stage2_mixed",
        "train_path": STAGE2_PATH,
        "learning_rate": 3e-5,
        "num_train_epochs": 5,
        "warmup_ratio": 0.05,
    },
    {
        "name": "stage3_real_only_low_lr",
        "train_path": STAGE3_PATH,
        "learning_rate": 1e-5,
        "num_train_epochs": 5,
        "warmup_ratio": 0.00,
    },
]


# ============================================================
# Cleaning / filtering
# ============================================================

def clean_text(text):
    text = str(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_example(ex):
    ex["source"] = clean_text(ex["source"])
    ex["target"] = clean_text(ex["target"])
    return ex


def valid_example(ex):
    return len(ex["source"]) > 0 and len(ex["target"]) > 0


def length_filter(ex, max_chars=1000, ratio_limit=3.0):
    src = ex["source"]
    tgt = ex["target"]

    if len(src) > max_chars or len(tgt) > max_chars:
        return False

    src_len = max(len(src.split()), 1)
    tgt_len = max(len(tgt.split()), 1)

    ratio = max(src_len / tgt_len, tgt_len / src_len)

    return ratio <= ratio_limit


def deduplicate_hf_dataset(ds):
    seen = set()
    keep_indices = []

    for i, ex in enumerate(ds):
        pair = (
            ex["src_lang"],
            ex["tgt_lang"],
            ex["source"],
            ex["target"],
        )

        if pair not in seen:
            seen.add(pair)
            keep_indices.append(i)

    return ds.select(keep_indices)


def normalize_lang_code(lang):
    lang = str(lang).strip()

    if lang not in LANG_MAP:
        raise ValueError(
            f"Unknown language code: {lang}. "
            f"Expected one of: {sorted(LANG_MAP.keys())}"
        )

    return LANG_MAP[lang]


def load_jsonl_dataset(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    print(f"Loading dataset from: {path}")

    df = pd.read_json(path, lines=True)

    required_cols = {"src_lang", "tgt_lang", "source", "target"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing columns in {path}: {missing_cols}")

    df = df[["src_lang", "tgt_lang", "source", "target"]].copy()

    df = df.dropna(subset=["src_lang", "tgt_lang", "source", "target"])

    for col in ["src_lang", "tgt_lang", "source", "target"]:
        df[col] = df[col].astype(str).str.strip()

    df = df[(df["source"] != "") & (df["target"] != "")]

    df["src_lang"] = df["src_lang"].apply(normalize_lang_code)
    df["tgt_lang"] = df["tgt_lang"].apply(normalize_lang_code)

    df = df.drop_duplicates(
        subset=["src_lang", "tgt_lang", "source", "target"],
        keep="first",
    ).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"No valid examples left after pandas cleaning: {path}")

    ds = Dataset.from_pandas(df, preserve_index=False)

    print("Raw examples after pandas cleaning:", len(ds))

    ds = ds.map(clean_example)
    ds = ds.filter(valid_example)
    ds = ds.filter(length_filter)
    ds = deduplicate_hf_dataset(ds)

    if len(ds) == 0:
        raise ValueError(f"No valid examples left after HF filtering: {path}")

    print("Clean examples:", len(ds))
    print("Columns:", ds.column_names)
    print("Sample:", ds[0])

    print("Language distribution:")
    print(pd.DataFrame(ds).groupby(["src_lang", "tgt_lang"]).size())

    return ds


def build_validation_dataset():
    print("Using real validation set:", VALIDATION_PATH)
    return load_jsonl_dataset(VALIDATION_PATH)


# ============================================================
# Load tokenizer and model
# ============================================================

print("Loading tokenizer and model:", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


# ============================================================
# Add Batak language token if needed
# ============================================================

if BBC_LANG not in tokenizer.get_vocab():
    print(f"Adding new language token: {BBC_LANG}")

    tokenizer.add_special_tokens({
        "additional_special_tokens": [BBC_LANG]
    })

    model.resize_token_embeddings(len(tokenizer))

    new_id = tokenizer.convert_tokens_to_ids(BBC_LANG)
    init_id = tokenizer.convert_tokens_to_ids(INIT_LANG)

    print("New Batak token id:", new_id)
    print("Init Indonesian token id:", init_id)

    with torch.no_grad():
        model.model.shared.weight[new_id] = model.model.shared.weight[init_id].clone()

else:
    print(f"{BBC_LANG} already exists in tokenizer vocab.")


# Important for gradient checkpointing
model.config.use_cache = False

# Important:
# Your data is mixed-direction:
# bbc_Latn -> ind_Latn
# bbc_Latn -> eng_Latn
# ind_Latn -> bbc_Latn
# eng_Latn -> bbc_Latn
#
# Therefore, do NOT force one global target language token.
model.config.forced_bos_token_id = None
model.generation_config.forced_bos_token_id = None


# ============================================================
# Tokenization
# ============================================================

def preprocess_function(examples):
    input_ids = []
    attention_masks = []
    labels = []

    for src_lang, tgt_lang, source, target in zip(
        examples["src_lang"],
        examples["tgt_lang"],
        examples["source"],
        examples["target"],
    ):
        src_lang_nllb = normalize_lang_code(src_lang)
        tgt_lang_nllb = normalize_lang_code(tgt_lang)

        tokenizer.src_lang = src_lang_nllb
        tokenizer.tgt_lang = tgt_lang_nllb

        model_input = tokenizer(
            source,
            max_length=MAX_LENGTH,
            truncation=True,
        )

        label = tokenizer(
            text_target=target,
            max_length=MAX_LENGTH,
            truncation=True,
        )

        input_ids.append(model_input["input_ids"])
        attention_masks.append(model_input["attention_mask"])
        labels.append(label["input_ids"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }


def tokenize_dataset(ds):
    return ds.map(
        preprocess_function,
        batched=True,
        remove_columns=ds.column_names,
    )


data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(eval_preds) -> dict[str, float]:
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

    preds = preds.astype(np.int64).tolist()
    labels = labels.astype(np.int64).tolist()

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    chrf = CHRF(word_order=2)
    bleu = BLEU(tokenize="flores200")

    return {
        "chrF++": round(chrf.corpus_score(decoded_preds, [decoded_labels]).score, 2),
        "BLEU": round(bleu.corpus_score(decoded_preds, [decoded_labels]).score, 2),
    }


# ============================================================
# Training helper
# ============================================================

bf16_available = torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def make_training_args(stage):
    stage_output_dir = f"{OUTPUT_DIR}/{stage['name']}"

    return Seq2SeqTrainingArguments(
        output_dir=stage_output_dir,

        learning_rate=stage["learning_rate"],
        warmup_ratio=stage["warmup_ratio"],

        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,

        num_train_epochs=stage["num_train_epochs"],

        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        generation_num_beams=1,

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,

        logging_steps=10,

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

        seed=SEED,
        data_seed=SEED,

        max_grad_norm=1.0,
    )


def cleanup_stage_checkpoints(stage_output_dir):
    stage_output_dir = Path(stage_output_dir)

    for path in stage_output_dir.glob("checkpoint-*"):
        print("Deleting checkpoint:", path)
        shutil.rmtree(path)


def run_stage(stage, tokenized_train, tokenized_validation):
    print("=" * 80)
    print(f"Starting {stage['name']}")
    print("=" * 80)

    training_args = make_training_args(stage)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    stage_save_dir = f"{OUTPUT_DIR}/{stage['name']}/final"

    print(f"Saving {stage['name']} model to:", stage_save_dir)

    trainer.save_model(stage_save_dir)
    tokenizer.save_pretrained(stage_save_dir)

    cleanup_stage_checkpoints(training_args.output_dir)

    print(f"Finished {stage['name']}")

    return trainer


# ============================================================
# Main curriculum
# ============================================================

print("=" * 80)
print("Preparing stable validation set")
print("=" * 80)

validation_ds = build_validation_dataset()
tokenized_validation = tokenize_dataset(validation_ds)

print("=" * 80)
print("Starting curriculum training")
print("=" * 80)

last_trainer = None

for stage in STAGES:
    train_ds = load_jsonl_dataset(stage["train_path"])
    tokenized_train = tokenize_dataset(train_ds)

    last_trainer = run_stage(
        stage=stage,
        tokenized_train=tokenized_train,
        tokenized_validation=tokenized_validation,
    )

print("=" * 80)
print("Saving final curriculum model")
print("=" * 80)

FINAL_DIR = f"{OUTPUT_DIR}/final_curriculum_model"

last_trainer.save_model(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

print("Final model saved to:", FINAL_DIR)

print("=" * 80)
print("Evaluating final model on real test set")
print("=" * 80)

test_ds = load_jsonl_dataset(TEST_PATH)
tokenized_test = tokenize_dataset(test_ds)

test_metrics = last_trainer.evaluate(
    eval_dataset=tokenized_test,
    metric_key_prefix="test",
)

print("Test metrics:", test_metrics)

print("Done.")