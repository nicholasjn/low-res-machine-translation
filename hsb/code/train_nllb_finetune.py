import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


BASE_DIR = Path(__file__).resolve().parent

DIRECTION_CONFIGS = {
    "de-hsb": {
        "output_dir": "../models/nllb-600M-de-hsb-finetuned",
        "src_lang": "deu_Latn",
        "tgt_lang": "hsb_Latn",
    },
    "hsb-de": {
        "output_dir": "../models/nllb-600M-hsb-de-finetuned",
        "src_lang": "hsb_Latn",
        "tgt_lang": "deu_Latn",
    },
}


def split_paths(split_dir: str, direction: str) -> dict[str, Path]:
    split_dir_obj = resolve_path(split_dir)
    if direction == "de-hsb":
        return {
            "train_src_path": split_dir_obj / "train.de",
            "train_tgt_path": split_dir_obj / "train.hsb",
            "dev_src_path": split_dir_obj / "dev.de",
            "dev_tgt_path": split_dir_obj / "dev.hsb",
        }
    return {
        "train_src_path": split_dir_obj / "train.hsb",
        "train_tgt_path": split_dir_obj / "train.de",
        "dev_src_path": split_dir_obj / "dev.hsb",
        "dev_tgt_path": split_dir_obj / "dev.de",
    }


def resolve_path(path: str) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return BASE_DIR / path_obj


def read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f]


def load_parallel_data(src_path: str, tgt_path: str, debug: bool) -> Dataset:
    src_path_obj = resolve_path(src_path)
    tgt_path_obj = resolve_path(tgt_path)
    sources = read_lines(src_path_obj)
    targets = read_lines(tgt_path_obj)

    if len(sources) != len(targets):
        raise ValueError(f"Source/target length mismatch: {len(sources)} vs {len(targets)}")

    if debug:
        sources = sources[:10]
        targets = targets[:10]

    print(f"Loaded {len(sources)} sentence pairs from {src_path_obj} and {tgt_path_obj}")
    return Dataset.from_dict({"source": sources, "target": targets})


def tokenize_dataset(
    dataset: Dataset, tokenizer, max_length: int, src_lang: str, tgt_lang: str
) -> Dataset:
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    src_id = tokenizer.convert_tokens_to_ids(src_lang)
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if src_id == tokenizer.unk_token_id:
        raise RuntimeError(f"{src_lang} maps to unk_token_id. Check the tokenizer/model.")
    if tgt_id == tokenizer.unk_token_id:
        raise RuntimeError(f"{tgt_lang} maps to unk_token_id. Check the tokenizer/model.")

    def preprocess(batch):
        return tokenizer(
            batch["source"],
            text_target=batch["target"],
            max_length=max_length,
            truncation=True,
        )

    return dataset.map(preprocess, batched=True, remove_columns=["source", "target"])


def clear_generation_params_from_model_config(model) -> None:
    if hasattr(model.config, "forced_bos_token_id"):
        model.config.forced_bos_token_id = None


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune extended NLLB translation model.")
    parser.add_argument("--direction", choices=["de-hsb", "hsb-de"], default="de-hsb")
    parser.add_argument("--model_path", default="../models/nllb-600M-hsb-init")
    parser.add_argument("--split_dir", default="../data/splits")
    parser.add_argument("--train_src_path", default=None)
    parser.add_argument("--train_tgt_path", default=None)
    parser.add_argument("--dev_src_path", default=None)
    parser.add_argument("--dev_tgt_path", default=None)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use only the first 10 train/dev pairs and tiny training settings.",
    )
    parser.add_argument("--num_train_epochs", type=float, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    direction_config = DIRECTION_CONFIGS[args.direction]
    paths = split_paths(args.split_dir, args.direction)
    train_src_path = args.train_src_path or paths["train_src_path"]
    train_tgt_path = args.train_tgt_path or paths["train_tgt_path"]
    dev_src_path = args.dev_src_path or paths["dev_src_path"]
    dev_tgt_path = args.dev_tgt_path or paths["dev_tgt_path"]
    output_dir = resolve_path(args.output_dir or direction_config["output_dir"])
    src_lang = direction_config["src_lang"]
    tgt_lang = direction_config["tgt_lang"]

    resolve_path("../outputs").mkdir(parents=True, exist_ok=True)
    resolve_path("../models").mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        print("DEBUG mode is ON: using first 10 train/dev pairs, 1 epoch, batch size 1.")
        args.num_train_epochs = 1
        args.per_device_train_batch_size = 1
        args.per_device_eval_batch_size = 1
        args.gradient_accumulation_steps = 1
    else:
        print("DEBUG mode is OFF: using full train/dev data.")

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. CPU debug is allowed, but training will be slow.")

    print("\nLoading tokenizer and model...")
    model_path = resolve_path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    hsb_id = tokenizer.convert_tokens_to_ids("hsb_Latn")
    print("hsb_Latn token id:", hsb_id)
    print("unk token id:", tokenizer.unk_token_id)
    if hsb_id == tokenizer.unk_token_id:
        raise RuntimeError("hsb_Latn maps to unk_token_id. Run CODE/extend_nllb_hsb.py first.")

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    print("Direction:", args.direction)
    print("Source language:", src_lang)
    print("Target language:", tgt_lang)
    print("forced_bos_token_id:", forced_bos_token_id)
    if forced_bos_token_id == tokenizer.unk_token_id:
        raise RuntimeError(f"{tgt_lang} maps to unk_token_id. Check the tokenizer/model.")

    model.generation_config.forced_bos_token_id = forced_bos_token_id
    clear_generation_params_from_model_config(model)

    train_dataset = load_parallel_data(train_src_path, train_tgt_path, args.debug)
    dev_dataset = load_parallel_data(dev_src_path, dev_tgt_path, args.debug)
    train_dataset = tokenize_dataset(
        train_dataset, tokenizer, args.max_length, src_lang, tgt_lang
    )
    dev_dataset = tokenize_dataset(
        dev_dataset, tokenizer, args.max_length, src_lang, tgt_lang
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=1 if args.debug else 50,
        save_strategy="no",
        predict_with_generate=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
    )

    print("\nStarting NLLB fine-tuning...")
    trainer.train()

    print("\nSaving fine-tuned model and tokenizer...")
    clear_generation_params_from_model_config(model)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()
