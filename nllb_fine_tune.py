# Load Dataset
import pandas as pd
from datasets import Dataset, DatasetDict

NUSAX_TRAIN_URL = "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/mt/train.csv"
NUSAX_VALID_URL = "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/mt/valid.csv"
SEED = 42

train_df = pd.read_csv(NUSAX_TRAIN_URL)
valid_df = pd.read_csv(NUSAX_VALID_URL)

COL_BBC = "toba_batak"
COL_EN = "english"
COL_ID = "indonesian"

train_df = train_df[[COL_EN, COL_BBC]].dropna().drop_duplicates()
valid_df = valid_df[[COL_EN, COL_BBC]].dropna().drop_duplicates()

train_df = train_df.rename(columns={
    COL_EN: "source",
    COL_BBC: "target"
})

valid_df = valid_df.rename(columns={
    COL_EN: "source",
    COL_BBC: "target"
})

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df, preserve_index=False),
    "validation": Dataset.from_pandas(valid_df, preserve_index=False)
})

print("Train examples:", len(dataset["train"]))
print("Validation examples:", len(dataset["validation"]))
print(dataset["train"][0])

import re
# Clean Text
def clean_text(ex):
    ex = str(ex)
    ex = ex.strip()
    ex = re.sub(r"\s+", " ", ex)
    return ex

def clean_example(ex):
    ex["source"] = clean_text(ex["source"])
    ex["target"] = clean_text(ex["source"])

dataset = dataset.map(clean_example)

# Filter empty example
def valid_example(ex):
    return len(ex["source"]) > 0 and len(ex["target"]) > 0

dataset = dataset.filter(valid_example)

# Filter long sentence
def length_filter(ex, max_chars=1000, ratio_limit=3.0):
    src = ex["source"]
    tgt = ex["target"]

    if len(src) > max_chars or len(tgt) > max_chars:
        return False

    src_len = max(len(src.split()),1)
    tgt_len = max(len(tgt.split()),1)
    ratio = max(src_len / tgt_len, tgt_len / src_len)
    return ratio <= ratio_limit

dataset = dataset.filter(length_filter)

# remove exact duplicate
def deduplicate_split(ex):
    seen = set()
    keep = []
    for x in ex:
        pair = (x["source"], x["target"])
        if pair not in seen:
            seen.add(pair)
            keep.append(x)
    return keep

dataset["train"] = deduplicate_split(dataset["train"])
dataset["validation"] = deduplicate_split(dataset["validation"])

dataset = DatasetDict({
    "train": Dataset.from_list(dataset["train"]),
    "validation": Dataset.from_list(dataset["validation"])
})

# print(dataset)
print("="*80)
print("Train examples:", len(dataset["train"]))
print("Validation examples:", len(dataset["validation"]))
print(dataset["train"]

# Load NLLB-200-600M

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
import torch

MODEL_NAME = "facebook/nllb-200-distilled-600M"
PROJECT_DIR = "/content/drive/MyDrive/BatakNLLB"
OUTPUT_DIR = f"{PROJECT_DIR}/nllb-600m-batak-20epoch"

SRC_LANG = "eng_Latn"
TGT_LANG = "bbc_Latn"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


# Check target language tag exist or not in model tokenizer
if TGT_LANG not in tokenizer.get_vocab():
    # Add new language tag for Batak
    NEW_LANG_TAG = "bbc_Latn"
    INIT_LANG = "ind_Latn"

    tokenizer.add_special_tokens({"additional_special_tokens":[NEW_LANG_TAG]})

    model.resize_token_embeddings(len(tokenizer))

    new_id = tokenizer.convert_tokens_to_ids(NEW_LANG_TAG)
    init_id = tokenizer.convert_tokens_to_ids(INIT_LANG)

    with torch.no_grad():
        model.model.shared.weight[new_id] = model.model.shared.weight[init_id].clone()

# Tokenization function
MAX_LENGTH = 256

tokenizer.src_lang = SRC_LANG
tokenizer.tgt_lang = TGT_LANG

def preprocess_function(examples):
    inputs = examples["source"]
    targets = examples["target"]

    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)

    labels = tokenizer(text_target=targets, max_length=MAX_LENGTH, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Data Collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Metrics evaluation using chrF and BLEU
from sacrebleu.metrics import BLEU, CHRF
import numpy as np

def compute_metrics(eval_preds) -> dict[str, float]:
    """Compute chrF++, BLEU, and COMET.

    Args:
        eval_preds: Model output in token from Seq2SeqTrainer.

    Returns:
        Dict with keys "chrF++", "spBLEU".
    """

    preds, labels = eval_preds

    # Sometimes Hugging Face returns predictions as a tuple
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.asarray(preds)
    labels = np.asarray(labels)

    pad_id = tokenizer.pad_token_id
    vocab_size = len(tokenizer)

    # Change -100 in labels as padding token
    # Replace invalid label ids

    labels = np.where(labels == -100, pad_id, labels)

    # Extra safety for predictions and labels
    preds = np.where(preds == -100, pad_id, preds)

    preds = np.where(preds < 0, pad_id, preds)
    labels = np.where(labels < 0, pad_id, labels)
    preds = np.where(preds >= vocab_size, pad_id, preds)
    labels = np.where(labels >= vocab_size, pad_id, labels)

    # Convert to normal Python int lists for tokenizer safety
    preds = preds.astype(np.int64).tolist()
    labels = labels.astype(np.int64).tolist()

    # Decode prediction and label token become text
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    chrf = CHRF(word_order=2)  # word_order=2 → chrF++
    bleu = BLEU(tokenize="flores200")  # SPM tokeniser matching NLLB paper
    # bleu = BLEU(tokenize="13a")  # Use this for standard BLEU

    scores: dict[str, float] = {
        "chrF++": round(chrf.corpus_score(decoded_preds, [decoded_labels]).score, 2),
        "BLEU": round(bleu.corpus_score(decoded_preds, [decoded_labels]).score, 2),
    }
    return scores

# Training arguments

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    learning_rate=3e-5,
    warmup_ratio=0.10, # paper used 15000

    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,

    num_train_epochs=10, # Paper use 10-15 epochs depending on language

    predict_with_generate=True,
    generation_max_length=256,
    generation_num_beams=1,

    eval_strategy="epoch",
    # eval_steps=50,

    save_strategy="no",
    # save_steps=50,
    # save_total_limit=1,

    logging_steps=10,

    load_best_model_at_end=False,
    # metric_for_best_model="chrF++",
    # greater_is_better=True,

    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),

    report_to="none"
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Training and save model
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Check metrics after training
trainer.args.generation_num_beams = 4
trainer.args.generation_max_length = 256

final_metrics = trainer.evaluate()
print(final_metrics)
