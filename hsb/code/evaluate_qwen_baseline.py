import argparse
from pathlib import Path

import torch
from sacrebleu import corpus_bleu, corpus_chrf
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent

DIRECTION_CONFIGS = {
    "de-hsb": {
        "src_path": "../data/splits/test.de",
        "ref_path": "../data/splits/test.hsb",
        "output_path": "../outputs/qwen_de-hsb_predictions.txt",
        "source_label": "German",
        "target_label": "Upper Sorbian",
        "instruction": "Translate the following German sentence into Upper Sorbian.",
    },
    "hsb-de": {
        "src_path": "../data/splits/test.hsb",
        "ref_path": "../data/splits/test.de",
        "output_path": "../outputs/qwen_hsb-de_predictions.txt",
        "source_label": "Upper Sorbian",
        "target_label": "German",
        "instruction": "Translate the following Upper Sorbian sentence into German.",
    },
}


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


def load_model(model_name: str):
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Qwen 7B may not fit or run practically on CPU.")

    print("\nLoading Qwen model...")
    print("Model:", model_name)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower() or "cuda" in message.lower():
            raise RuntimeError(
                "Failed to load Qwen. This is likely a GPU memory/device issue. "
                "Try a smaller --model_name, reduce other GPU jobs, or run on an A100 node."
            ) from exc
        raise
    except Exception as exc:
        raise RuntimeError(
            "Failed to load Qwen. Check that the model is accessible, dependencies are installed, "
            "and the local GPU has enough memory."
        ) from exc

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    print("Qwen loaded")
    print("Device map:", getattr(model, "hf_device_map", "single device"))

    return tokenizer, model


def extract_translation(decoded_text: str, prompt: str, target_label: str, source_label: str) -> str:
    text = decoded_text.strip()
    target_marker = f"{target_label}:"
    source_marker = f"{source_label}:"

    if target_marker in text:
        text = text.rsplit(target_marker, 1)[-1].strip()
    else:
        text = text.replace(prompt, "").strip()
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("Translate the following"):
                continue
            if line.startswith("Only output the translation"):
                continue
            if line.startswith(source_marker) or line.startswith(target_marker):
                continue
            lines.append(line)
        if lines:
            text = lines[-1]

    return text.strip().strip("\"'")


def translate_with_qwen(source: str, tokenizer, model, direction_config: dict) -> str:
    source_label = direction_config["source_label"]
    target_label = direction_config["target_label"]
    prompt = f"""{direction_config["instruction"]}
Only output the translation.

{source_label}: {source}
{target_label}:"""

    model_device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(model_device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return extract_translation(decoded, prompt, target_label, source_label)


def evaluate(tokenizer, model, src_sentences: list[str], references: list[str], output_path: str, direction_config: dict):
    predictions = []
    test_size = len(src_sentences)

    print(f"\nRunning Qwen baseline on {test_size} sentences...")
    for i, source in enumerate(src_sentences):
        prediction = translate_with_qwen(source, tokenizer, model, direction_config)
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
    parser = argparse.ArgumentParser(description="Evaluate Qwen translation baseline.")
    parser.add_argument("--direction", choices=["de-hsb", "hsb-de"], default="de-hsb")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num_test", type=int, default=2000)
    parser.add_argument("--src_path", default=None)
    parser.add_argument("--ref_path", default=None)
    parser.add_argument("--output_path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    resolve_path("../outputs").mkdir(parents=True, exist_ok=True)
    resolve_path("../models").mkdir(parents=True, exist_ok=True)
    resolve_path("../data/splits").mkdir(parents=True, exist_ok=True)

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))

    direction_config = DIRECTION_CONFIGS[args.direction]
    src_path = args.src_path or direction_config["src_path"]
    ref_path = args.ref_path or direction_config["ref_path"]
    output_path = args.output_path or direction_config["output_path"]

    print("Direction:", args.direction)
    src_sentences, references = load_dataset(src_path, ref_path, args.num_test)
    tokenizer, model = load_model(args.model_name)
    evaluate(tokenizer, model, src_sentences, references, output_path, direction_config)


if __name__ == "__main__":
    main()
