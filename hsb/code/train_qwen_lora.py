import argparse
import inspect
import os
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


def str2bool(x):
    if isinstance(x, bool):
        return x
    return str(x).lower() in ["true", "1", "yes", "y"]


def infer_lang(path: str) -> str:
    name = os.path.basename(path)
    if name.endswith(".de"):
        return "German"
    if name.endswith(".hsb"):
        return "Upper Sorbian"
    return "target language"


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def build_prompt(src_text: str, src_lang: str, tgt_lang: str) -> str:
    return (
        "You are a translation system. Translate exactly and only output the translation. "
        "Do not explain.\n\n"
        f"Translate from {src_lang} to {tgt_lang}:\n{src_text}\n\n"
        "Translation:"
    )


class TranslationDataset(Dataset):
    def __init__(self, src_path, tgt_path, tokenizer, max_length, src_lang, tgt_lang, debug=False):
        src_lines = read_lines(src_path)
        tgt_lines = read_lines(tgt_path)

        n = min(len(src_lines), len(tgt_lines))
        src_lines = src_lines[:n]
        tgt_lines = tgt_lines[:n]

        if debug:
            print("DEBUG mode is ON: using first 10 sentence pairs.")
            src_lines = src_lines[:10]
            tgt_lines = tgt_lines[:10]

        self.pairs = list(zip(src_lines, tgt_lines))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        print(f"Loaded {len(self.pairs)} sentence pairs from {src_path} and {tgt_path}")

    def __len__(self):
        return len(self.pairs)

    def _encode(self, text: str):
        ids = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        return list(ids)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]

        prompt = build_prompt(src, self.src_lang, self.tgt_lang)
        full_text = prompt + " " + tgt + self.tokenizer.eos_token

        prompt_ids = self._encode(prompt)
        full_ids = self._encode(full_text)

        labels = full_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }


@dataclass
class CausalLMCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)
        pad_id = self.tokenizer.pad_token_id

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for x in features:
            input_ids = x["input_ids"]
            attention_mask = x["attention_mask"]
            labels = x["labels"]

            pad_len = max_len - len(input_ids)

            batch_input_ids.append(input_ids + [pad_id] * pad_len)
            batch_attention_mask.append(attention_mask + [0] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }



def load_qwen_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.load_in_4bit:
        import bitsandbytes  # noqa: F401

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return tokenizer, model


def make_training_args(args):
    kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=10,
        save_strategy="no",
        fp16=True,
        bf16=False,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
    )

    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_src_path", type=str, required=True)
    parser.add_argument("--train_tgt_path", type=str, required=True)
    parser.add_argument("--dev_src_path", type=str, required=True)
    parser.add_argument("--dev_tgt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--load_in_4bit", type=str2bool, default=True)
    parser.add_argument("--debug", type=str2bool, default=False)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    src_lang = infer_lang(args.train_src_path)
    tgt_lang = infer_lang(args.train_tgt_path)

    print("Model:", args.model_name)
    print("Direction:", src_lang, "->", tgt_lang)
    print("Output dir:", args.output_dir)
    print("Debug:", args.debug)
    print("4bit:", args.load_in_4bit)

    tokenizer, model = load_qwen_model(args)

    train_dataset = TranslationDataset(
        args.train_src_path,
        args.train_tgt_path,
        tokenizer,
        args.max_length,
        src_lang,
        tgt_lang,
        debug=args.debug,
    )

    eval_dataset = TranslationDataset(
        args.dev_src_path,
        args.dev_tgt_path,
        tokenizer,
        args.max_length,
        src_lang,
        tgt_lang,
        debug=args.debug,
    )

    training_args = make_training_args(args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalLMCollator(tokenizer),
    )

    trainer.train()

    print("Saving LoRA adapter to:", args.output_dir)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
