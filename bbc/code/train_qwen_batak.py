import os
from datasets import load_dataset, concatenate_datasets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling, default_data_collator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import math

RERUN = True

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
PROJECT_DIR = os.environ.get("PROJECT_DIR", "/dss/dsshome1/0C/go93pec2/batak-qwen")
DATA_DIR = f"{PROJECT_DIR}/data"
WIKIPEDIA_BBC = f"{DATA_DIR}/wikipedia_bbc.jsonl"
OUTPUT_DIR = f"{PROJECT_DIR}/outputs/qwen2.5-1.5b-batak-qlora"
CACHE_DIR = f"{PROJECT_DIR}/hf_cache"

SEED = 42

def reload_train_valid():
    dataset = load_dataset("json", data_files={
        "train":f"{DATA_DIR}/train.jsonl",
        "validation":f"{DATA_DIR}/valid.jsonl"
    }, cache_dir=CACHE_DIR)
    train_raw = dataset["train"]
    valid_raw = dataset["validation"]
    return train_raw, valid_raw
    

def load_madlad400():
    url = "https://huggingface.co/datasets/allenai/MADLAD-400/resolve/main/data/bbc/bbc_clean_0000.jsonl.gz"
    raw = load_dataset("json", data_files = {"train":url}, split="train", cache_dir=CACHE_DIR)
    raw = raw.filter(lambda x: x["text"] is not None and x["text"].strip() != "")
    return raw

def load_wikipedia():
    raw = load_dataset("json", data_files=WIKIPEDIA_BBC, split="train", cache_dir=CACHE_DIR)
    raw = raw.filter(lambda x: x["text"] is not None and x["text"].strip() != "")
    return raw

if not RERUN:
    madlad_raw = load_madlad400()
    madlad_raw = madlad_raw.select_columns(["text"])
    wiki_raw = load_wikipedia()
    wiki_raw = wiki_raw.select_columns(["text"])

    raw = concatenate_datasets([madlad_raw, wiki_raw])
    raw = raw.shuffle(seed=SEED)

    split = raw.train_test_split(test_size=0.05, seed=42, shuffle=True)

    train_raw = split["train"]
    valid_raw = split["test"]

    train_raw.to_json(f"{DATA_DIR}/train.jsonl", force_ascii=False)
    valid_raw.to_json(f"{DATA_DIR}/valid.jsonl", force_ascii=False)
else:
    train_raw, valid_raw = reload_train_valid()

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, cache_dir=CACHE_DIR)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
    texts = [t.strip() + tokenizer.eos_token for t in example["text"]]
    return tokenizer(texts, add_special_tokens=False)

train_tokenized = train_raw.map(
    tokenize_function,
    batched=True,
    remove_columns=train_raw.column_names,
    desc = "Tokenizing train"
)

valid_tokenized = valid_raw.map(
    tokenize_function,
    batched=True,
    remove_columns=valid_raw.column_names,
    desc = "Tokenizing valid"
)

# Chunking
seq_len = 1024
def group_texts(examples):
    concatenated = []
    for ids in examples["input_ids"]:
        concatenated.extend(ids)

    total_length = (len(concatenated) // seq_len) * seq_len

    input_ids = [concatenated[i : i + seq_len] for i in range(0, total_length, seq_len)]

    return {
        "input_ids": input_ids,
        "attention_mask": [[1] * seq_len for _ in input_ids],
        "labels": [x.copy() for x in input_ids],
    }

train_dataset = train_tokenized.map(
    group_texts,
    batched = True,
    desc = f"Grouping train into {seq_len}"
)

valid_dataset = valid_tokenized.map(
    group_texts,
    batched = True,
    desc = f"Grouping valid into {seq_len}"
)

print(train_dataset)
print(valid_dataset)

print("Train chunks:", len(train_dataset))
print("Valid chunks:", len(valid_dataset))
print("Approx train tokens:", len(train_dataset) * seq_len)
print("Approx valid tokens:", len(valid_dataset) * seq_len)



compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# 4-bit config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir=CACHE_DIR
)

model.config.use_cache=False

# LoRA config
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# model = PeftModel.from_pretrained(model, PREVIOUS_ADAPTER_DIR, is_trainable=True)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Check Steps
per_device_train_batch_size = 4
gradient_accumulation_steps = 8

steps_per_epoch = math.ceil(len(train_dataset) / (per_device_train_batch_size * gradient_accumulation_steps))

print("Steps per epoch:", steps_per_epoch)
print("Half epoch steps:", steps_per_epoch // 2)

# Training Config
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=4,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,

    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    weight_decay=0.0,
    max_grad_norm=1.0,

    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,

    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,

    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    optim="paged_adamw_8bit",
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
)

data_collator = default_data_collator

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator
)

# Training
last_checkpoint = None

if os.path.isdir(OUTPUT_DIR):
    checkpoints = [
        os.path.join(OUTPUT_DIR, d)
        for d in os.listdir(OUTPUT_DIR)
        if d.startswith("checkpoint-")
    ]

    if checkpoints:
        last_checkpoint = max(
            checkpoints,
            key=lambda x: int(x.split("-")[-1])
        )

if last_checkpoint:
    print("Resuming from checkpoint:", last_checkpoint)
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Saved to:", OUTPUT_DIR)
