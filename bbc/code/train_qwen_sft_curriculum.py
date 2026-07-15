from pathlib import Path
import shutil

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# ============================================================
# Config
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

PREV_ADAPTER = Path(
    "/dss/dsshome1/0C/go93pec2/batak-qwen/outputs/qwen2.5-1.5b-batak-qlora"
)

DATA_DIR = Path("data")

TRAIN_STAGE1_PATH = DATA_DIR / "train_stage1_synthetic_chat.jsonl"
TRAIN_STAGE2_PATH = DATA_DIR / "train_stage2_mixed_chat.jsonl"
TRAIN_STAGE3_PATH = DATA_DIR / "train_stage3_real_only_chat.jsonl"

VALID_PATH = DATA_DIR / "valid_real_chat.jsonl"
TEST_PATH = DATA_DIR / "test_real_chat.jsonl"

OUTPUT_ROOT = Path(
    "/dss/dsshome1/0C/go93pec2/batak-qwen-sft/outputs/qwen2.5-1.5b-batak-qlora-sft-curriculum"
)

SEED = 42
MAX_LENGTH = 768


# ============================================================
# Curriculum hyperparameters
# ============================================================

STAGES = [
    {
        "name": "stage1_synthetic",
        "train_path": TRAIN_STAGE1_PATH,
        "num_train_epochs": 4,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.05,
        "save_steps": 200,
        "eval_steps": 200,
    },
    {
        "name": "stage2_mixed",
        "train_path": TRAIN_STAGE2_PATH,
        "num_train_epochs": 4,
        "learning_rate": 1e-4,
        "warmup_ratio": 0.03,
        "save_steps": 200,
        "eval_steps": 200,
    },
    {
        "name": "stage3_real_only_low_lr",
        "train_path": TRAIN_STAGE3_PATH,
        "num_train_epochs": 3,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.00,
        "save_steps": 50,
        "eval_steps": 50,
    },
]

# ============================================================
# Utilities
# ============================================================

def check_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def load_json_dataset(path: Path):
    check_file(path)

    print("=" * 80)
    print("Loading dataset:", path)
    print("=" * 80)

    ds = load_dataset("json", data_files=str(path), split="train")

    if "messages" not in ds.column_names:
        raise ValueError(
            f"Dataset {path} must contain a 'messages' column. "
            f"Found columns: {ds.column_names}"
        )

    print("Examples:", len(ds))
    print("Columns:", ds.column_names)
    print("Sample:", ds[0])

    return ds


def format_dataset(ds, tokenizer):
    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    ds = ds.map(format_example)

    return ds


def cleanup_checkpoints(stage_output_dir: Path):
    for path in stage_output_dir.glob("checkpoint-*"):
        print("Deleting checkpoint:", path)
        shutil.rmtree(path)


# ============================================================
# Load tokenizer
# ============================================================

print("=" * 80)
print("Loading tokenizer")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


# ============================================================
# Load validation and test data
# ============================================================

valid_ds = load_json_dataset(VALID_PATH)
test_ds = load_json_dataset(TEST_PATH)

valid_ds = format_dataset(valid_ds, tokenizer)
test_ds = format_dataset(test_ds, tokenizer)


# ============================================================
# Load model in 4-bit and previous adapter
# ============================================================

print("=" * 80)
print("Loading base model and previous QLoRA adapter")
print("=" * 80)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False

model = prepare_model_for_kbit_training(model)

model = PeftModel.from_pretrained(
    model,
    PREV_ADAPTER,
    is_trainable=True,
)

model.print_trainable_parameters()

# ============================================================
# Training config helper
# ============================================================

def make_sft_config(stage):
    stage_output_dir = OUTPUT_ROOT / stage["name"]

    return SFTConfig(
        output_dir=str(stage_output_dir),

        num_train_epochs=stage["num_train_epochs"],

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,

        learning_rate=stage["learning_rate"],
        warmup_ratio=stage["warmup_ratio"],
        lr_scheduler_type="cosine",

        logging_steps=10,

        eval_strategy="steps",
        eval_steps=stage["eval_steps"],

        save_strategy="steps",
        save_steps=stage["save_steps"],
        save_total_limit=1,

        max_length=MAX_LENGTH,
        packing=False,

        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),

        optim="paged_adamw_8bit",

        gradient_checkpointing=True,
        max_grad_norm=1.0,

        remove_unused_columns=False,

        report_to="none",

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        seed=SEED,
        data_seed=SEED,
    )


# ============================================================
# Stage runner
# ============================================================

def run_stage(stage):
    print("=" * 80)
    print("Starting:", stage["name"])
    print("Train file:", stage["train_path"])
    print("Learning rate:", stage["learning_rate"])
    print("Epochs:", stage["num_train_epochs"])
    print("=" * 80)

    train_ds = load_json_dataset(stage["train_path"])
    train_ds = format_dataset(train_ds, tokenizer)

    training_args = make_sft_config(stage)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    stage_final_dir = OUTPUT_ROOT / stage["name"] / "final"

    print("Saving adapter for stage:", stage["name"])
    print("Save path:", stage_final_dir)

    trainer.save_model(str(stage_final_dir))
    tokenizer.save_pretrained(str(stage_final_dir))

    cleanup_checkpoints(Path(training_args.output_dir))

    print("Finished:", stage["name"])

    return trainer


# ============================================================
# Run curriculum
# ============================================================

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

last_trainer = None

for stage in STAGES:
    last_trainer = run_stage(stage)


# ============================================================
# Save final curriculum adapter
# ============================================================

FINAL_DIR = OUTPUT_ROOT / "final_curriculum_adapter"

print("=" * 80)
print("Saving final curriculum adapter")
print("=" * 80)

last_trainer.save_model(str(FINAL_DIR))
tokenizer.save_pretrained(str(FINAL_DIR))

print("Final adapter saved to:", FINAL_DIR)


# ============================================================
# Final test evaluation
# ============================================================

print("=" * 80)
print("Evaluating final model on test set")
print("=" * 80)

test_metrics = last_trainer.evaluate(
    eval_dataset=test_ds,
    metric_key_prefix="test",
)

print("Test metrics:")
print(test_metrics)

print("Done.")