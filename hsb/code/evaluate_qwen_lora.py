"""Evaluate a Qwen LoRA adapter on a parallel MT test set.

The script mirrors the baseline Qwen evaluator but loads a PEFT adapter on top
of Qwen/Qwen2.5-7B-Instruct by default. It prints BLEU/chrF++ and writes one
prediction per input line.
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm
import sacrebleu

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def str2bool(x):
    if isinstance(x, bool):
        return x
    return str(x).lower() in ["true", "1", "yes", "y"]


def infer_lang(path):
    name = Path(path).name
    if name.endswith(".de"):
        return "German"
    if name.endswith(".hsb"):
        return "Upper Sorbian"
    return "target language"


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [x.rstrip("\n") for x in f]


def build_prompt(src_text, src_lang, tgt_lang):
    return (
        "You are a translation system. Translate exactly and only output the translation. "
        "Do not explain.\n\n"
        f"Translate from {src_lang} to {tgt_lang}:\n{src_text}\n\n"
        "Translation:"
    )


def load_model(base_model, adapter_path, load_in_4bit=True):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate_one(model, tokenizer, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--src_path", type=str, required=True)
    parser.add_argument("--ref_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--direction", choices=["de-hsb", "hsb-de"], default=None)
    parser.add_argument("--num_test", type=int, default=2000)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--load_in_4bit", type=str2bool, default=True)
    args = parser.parse_args()

    if args.direction == "de-hsb":
        src_lang, tgt_lang = "German", "Upper Sorbian"
    elif args.direction == "hsb-de":
        src_lang, tgt_lang = "Upper Sorbian", "German"
    else:
        src_lang = infer_lang(args.src_path)
        tgt_lang = infer_lang(args.ref_path)

    src_all = read_lines(args.src_path)
    ref_all = read_lines(args.ref_path)
    if len(src_all) != len(ref_all):
        raise ValueError(f"Source/reference length mismatch: {len(src_all)} vs {len(ref_all)}")

    test_size = min(args.num_test, len(src_all))
    src_lines = src_all[:test_size]
    ref_lines = ref_all[:test_size]

    print("Base model:", args.base_model)
    print("Adapter:", args.adapter_path)
    print("Direction:", src_lang, "->", tgt_lang)
    print("Source:", args.src_path)
    print("Reference:", args.ref_path)
    print("Available pairs:", len(src_all))
    print("Evaluation pairs:", len(src_lines))

    tokenizer, model = load_model(args.base_model, args.adapter_path, args.load_in_4bit)

    preds = []
    for src in tqdm(src_lines):
        prompt = build_prompt(src, src_lang, tgt_lang)
        pred = generate_one(model, tokenizer, prompt, args.max_new_tokens)
        preds.append(pred)

    output_path = Path(args.output_path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(p.replace("\n", " ").strip() + "\n")

    bleu = sacrebleu.corpus_bleu(preds, [ref_lines])
    chrf = sacrebleu.corpus_chrf(preds, [ref_lines], word_order=2)

    print("BLEU:", bleu.score)
    print("chrF++:", chrf.score)
    print("Predictions saved to:", output_path)


if __name__ == "__main__":
    main()
