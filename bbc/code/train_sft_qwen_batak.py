from pathlib import Path
from datasets import load_from_disk, load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
PREV_ADAPTER = Path("/dss/dsshome1/0C/go93pec2/batak-qwen/outputs/qwen2.5-1.5b-batak-qlora")
INPUT_PATH = Path("data/sft_data.jsonl")
OUTPUT_PATH = Path("/dss/dsshome1/0C/go93pec2/batak-qwen-sft/outputs/qwen2.5-1.5b-batak-qlora-sft")
SEED = 42

dataset = load_dataset("json", data_files=str(INPUT_PATH), split="train")
dataset = dataset.train_test_split(test_size=0.05, seed=SEED)

train_ds = dataset["train"]
eval_ds = dataset["test"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Format with Qwen Chat template
def format_example(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

train_ds = train_ds.map(format_example)
eval_ds = eval_ds.map(format_example)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
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

# Training Config
training_args = SFTConfig(
    output_dir=OUTPUT_PATH,
    num_train_epochs=4,

    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,

    learning_rate=1e-4,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",

    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,

    max_length=768,
    packing=False,

    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),

    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    remove_unused_columns=False,

    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

# Trainer
trainer_sft = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
)

# Run Trainer
trainer_sft.train()
trainer_sft.save_model(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print("Saved to:", OUTPUT_PATH)
