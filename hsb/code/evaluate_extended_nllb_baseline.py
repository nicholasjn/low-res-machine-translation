import argparse
from pathlib import Path

import torch
from sacrebleu import corpus_bleu, corpus_chrf
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent

DIRECTION_CONFIGS = {
    "de-hsb": {
        "output_path": "../outputs/extended_nllb_de-hsb_predictions.txt",
        "src_lang": "deu_Latn",
        "tgt_lang": "hsb_Latn",
    },
    "hsb-de": {
        "output_path": "../outputs/extended_nllb_hsb-de_predictions.txt",
        "src_lang": "hsb_Latn",
        "tgt_lang": "deu_Latn",
    },
}


def test_paths(split_dir: str, direction: str) -> tuple[Path, Path]:
    split_dir_obj = resolve_path(split_dir)
    if direction == "de-hsb":
        return split_dir_obj / "test.de", split_dir_obj / "test.hsb"
    return split_dir_obj / "test.hsb", split_dir_obj / "test.de"


def resolve_path(path: str) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return BASE_DIR / path_obj


def read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f]


def load_dataset(src_path: str, ref_path: str, num_test: int):
    src_path_obj = resolve_path(src_path)
    ref_path_obj = resolve_path(ref_path)
    src_sentences = read_lines(src_path_obj)
    references = read_lines(ref_path_obj)

    if len(src_sentences) != len(references):
        raise ValueError(f"Source/reference length mismatch: {len(src_sentences)} vs {len(references)}")

    test_size = min(num_test, len(src_sentences))
    print("\nDataset:")
    print("Source:", src_path_obj)
    print("Reference:", ref_path_obj)
    print("Available pairs:", len(src_sentences))
    print("Evaluation pairs:", test_size)

    return src_sentences[:test_size], references[:test_size]


def load_model(model_path: str, src_lang: str, tgt_lang: str):
    model_path_obj = resolve_path(model_path)
    print("\nLoading extended NLLB model...")
    print("Model path:", model_path_obj)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: CUDA is not available. Running NLLB evaluation on CPU debug mode.")

    tokenizer = AutoTokenizer.from_pretrained(model_path_obj)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path_obj).to(device)
    model.eval()

    hsb_id = tokenizer.convert_tokens_to_ids("hsb_Latn")
    print("hsb_Latn token id:", hsb_id)
    print("unk token id:", tokenizer.unk_token_id)

    if hsb_id == tokenizer.unk_token_id:
        raise RuntimeError(
            "hsb_Latn still maps to unk_token_id. Please run CODE/extend_nllb_hsb.py first."
        )

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    print("Source language:", src_lang)
    print("Target language:", tgt_lang)
    print("forced_bos_token_id:", forced_bos_token_id)

    if forced_bos_token_id == tokenizer.unk_token_id:
        raise RuntimeError(f"{tgt_lang} maps to unk_token_id. Check the tokenizer/model.")

    tokenizer.src_lang = src_lang
    return tokenizer, model, device, forced_bos_token_id


def translate_with_nllb(source: str, tokenizer, model, device, forced_bos_token_id: int) -> str:
    inputs = tokenizer(source, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            num_beams=5,
            max_new_tokens=128,
        )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()


def evaluate(tokenizer, model, device, forced_bos_token_id: int, src_sentences, references, output_path: str):
    predictions = []
    test_size = len(src_sentences)

    print(f"\nRunning extended NLLB baseline on {test_size} sentences...")
    for i, source in enumerate(src_sentences):
        prediction = translate_with_nllb(source, tokenizer, model, device, forced_bos_token_id)
        predictions.append(prediction)

        if i < 5:
            print("=" * 40)
            print("SRC:", source)
            print("REF:", references[i])
            print("PRED:", prediction)

        if (i + 1) % 100 == 0 or (i + 1) == test_size:
            print(f"[{i + 1}/{test_size}] done")

    output_path_obj = resolve_path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path_obj, "w", encoding="utf-8") as f:
        for prediction in predictions:
            f.write(prediction + "\n")

    bleu = corpus_bleu(predictions, [references])
    chrf = corpus_chrf(predictions, [references])

    print("\nFinal scores:")
    print("BLEU:", bleu.score)
    print("chrF++:", chrf.score)
    print("Predictions saved to:", output_path_obj)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate extended NLLB translation baseline.")
    parser.add_argument("--direction", choices=["de-hsb", "hsb-de"], default="de-hsb")
    parser.add_argument("--model_path", default="../models/nllb-600M-hsb-init")
    parser.add_argument("--split_dir", default="../data/splits")
    parser.add_argument("--src_path", default=None)
    parser.add_argument("--ref_path", default=None)
    parser.add_argument("--num_test", type=int, default=2000)
    parser.add_argument("--output_path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    resolve_path("../outputs").mkdir(parents=True, exist_ok=True)
    resolve_path("../models").mkdir(parents=True, exist_ok=True)

    direction_config = DIRECTION_CONFIGS[args.direction]
    default_src_path, default_ref_path = test_paths(args.split_dir, args.direction)
    src_path = args.src_path or default_src_path
    ref_path = args.ref_path or default_ref_path
    output_path = args.output_path or direction_config["output_path"]

    print("Direction:", args.direction)
    src_sentences, references = load_dataset(src_path, ref_path, args.num_test)
    tokenizer, model, device, forced_bos_token_id = load_model(
        args.model_path,
        direction_config["src_lang"],
        direction_config["tgt_lang"],
    )
    evaluate(tokenizer, model, device, forced_bos_token_id, src_sentences, references, output_path)


if __name__ == "__main__":
    main()
