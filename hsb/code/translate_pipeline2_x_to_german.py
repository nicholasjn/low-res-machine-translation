import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


TARGET_LANG = "deu_Latn"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate multilingual non-German source sentences into German for Pipeline 2."
    )
    parser.add_argument("--src_path", required=True, type=Path)
    parser.add_argument("--lang_path", required=True, type=Path)
    parser.add_argument("--hsb_path", required=True, type=Path)
    parser.add_argument("--output_de", required=True, type=Path)
    parser.add_argument("--output_hsb", required=True, type=Path)
    parser.add_argument("--output_tsv", required=True, type=Path)
    parser.add_argument("--model_name", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_lines(path):
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        return [line.rstrip("\n\r") for line in handle]


def check_outputs(paths, overwrite):
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise SystemExit(
            "Refusing to overwrite existing output file(s). "
            "Pass --overwrite to replace them:\n" + joined
        )


def load_examples(src_path, lang_path, hsb_path, num_examples):
    src_lines = read_lines(src_path)
    lang_lines = read_lines(lang_path)
    hsb_lines = read_lines(hsb_path)

    lengths = (len(src_lines), len(lang_lines), len(hsb_lines))
    if len(set(lengths)) != 1:
        raise SystemExit(
            f"Input line count mismatch: src={lengths[0]}, lang={lengths[1]}, hsb={lengths[2]}"
        )

    examples = []
    skipped_empty = 0
    for src_sentence, src_lang, hsb_sentence in zip(src_lines, lang_lines, hsb_lines):
        if not src_sentence.strip() or not src_lang.strip() or not hsb_sentence.strip():
            skipped_empty += 1
            continue
        examples.append(
            {
                "src_lang": src_lang.strip(),
                "src_sentence": src_sentence,
                "hsb_sentence": hsb_sentence,
            }
        )

    if num_examples > 0:
        examples = examples[:num_examples]

    print(f"input_lines\t{lengths[0]}")
    print(f"skipped_empty_aligned_rows\t{skipped_empty}")
    print(f"examples_to_translate\t{len(examples)}")
    return examples


def load_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"model_name\t{model_name}")
    print(f"device\t{device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TARGET_LANG)
    if forced_bos_token_id == tokenizer.unk_token_id:
        raise RuntimeError(f"{TARGET_LANG} maps to unk_token_id. Check the tokenizer/model.")

    print(f"target_lang\t{TARGET_LANG}")
    print(f"forced_bos_token_id\t{forced_bos_token_id}")
    return tokenizer, model, device, forced_bos_token_id


def translate_batch(sentences, src_lang, tokenizer, model, device, forced_bos_token_id, max_length):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=max_length,
        )

    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def translate_examples(examples, tokenizer, model, device, forced_bos_token_id, batch_size, max_length):
    grouped_indices = defaultdict(list)
    for index, example in enumerate(examples):
        grouped_indices[example["src_lang"]].append(index)

    translations = [""] * len(examples)
    with tqdm(total=len(examples), desc="Translating", unit="sent") as progress:
        for src_lang in sorted(grouped_indices):
            token_id = tokenizer.convert_tokens_to_ids(src_lang)
            if token_id == tokenizer.unk_token_id:
                raise RuntimeError(f"{src_lang} maps to unk_token_id. Check lang_path or model.")

            indices = grouped_indices[src_lang]
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                sentences = [examples[index]["src_sentence"] for index in batch_indices]
                batch_translations = translate_batch(
                    sentences,
                    src_lang,
                    tokenizer,
                    model,
                    device,
                    forced_bos_token_id,
                    max_length,
                )
                for index, translation in zip(batch_indices, batch_translations):
                    translations[index] = translation.strip()
                progress.update(len(batch_indices))

    return translations


def write_outputs(examples, translations, output_de, output_hsb, output_tsv):
    for path in (output_de, output_hsb, output_tsv):
        path.parent.mkdir(parents=True, exist_ok=True)

    with output_de.open("w", encoding="utf-8", newline="\n") as de_file, output_hsb.open(
        "w", encoding="utf-8", newline="\n"
    ) as hsb_file:
        for example, translation in zip(examples, translations):
            de_file.write(translation + "\n")
            hsb_file.write(example["hsb_sentence"] + "\n")

    with output_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["src_lang", "src_sentence", "mt_german", "hsb_sentence"])
        for example, translation in zip(examples, translations):
            writer.writerow(
                [
                    example["src_lang"],
                    example["src_sentence"],
                    translation,
                    example["hsb_sentence"],
                ]
            )


def print_summary(examples):
    counts = Counter(example["src_lang"] for example in examples)
    print("counts_per_language")
    for src_lang, count in sorted(counts.items()):
        print(f"{src_lang}\t{count}")
    print(f"total_translated_lines\t{len(examples)}")


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be > 0")
    if args.max_length <= 0:
        raise SystemExit("--max_length must be > 0")

    check_outputs([args.output_de, args.output_hsb, args.output_tsv], args.overwrite)
    examples = load_examples(args.src_path, args.lang_path, args.hsb_path, args.num_examples)
    tokenizer, model, device, forced_bos_token_id = load_model(args.model_name)
    translations = translate_examples(
        examples,
        tokenizer,
        model,
        device,
        forced_bos_token_id,
        args.batch_size,
        args.max_length,
    )
    write_outputs(examples, translations, args.output_de, args.output_hsb, args.output_tsv)
    print_summary(examples)
    print(f"output_de\t{args.output_de}")
    print(f"output_hsb\t{args.output_hsb}")
    print(f"output_tsv\t{args.output_tsv}")


if __name__ == "__main__":
    main()
