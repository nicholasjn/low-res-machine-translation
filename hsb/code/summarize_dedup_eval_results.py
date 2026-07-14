import glob
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
XCOMET_DIR = ROOT / "outputs" / "xcomet"
OUTPUT_PATH = ROOT / "outputs" / "dedup_eval_summary.tsv"


def infer_from_name(name):
    base = Path(name).name
    model = "pending"
    train_data = "pending"
    direction = "pending"
    test_set = "pending"

    if "qwen7b_lora" in base or "qwen_lora" in base:
        model = "qwen7b_lora"
    elif "nllb_cleaned_finetuned" in base:
        model = "nllb_finetuned"
        train_data = "cleaned_train"
    elif "nllb_finetuned" in base:
        model = "nllb_finetuned"
        train_data = "original_train"

    if train_data == "pending":
        if "dedup_vs_cleaned_train" in base:
            train_data = "cleaned_train"
        elif "dedup_vs_original_train" in base:
            train_data = "original_train"

    if "de-hsb" in base:
        direction = "de-hsb"
    elif "hsb-de" in base:
        direction = "hsb-de"

    if "dedup_vs_cleaned_train" in base:
        test_set = "original_test_dedup_vs_cleaned_train"
    elif "dedup_vs_original_train" in base:
        test_set = "original_test_dedup_vs_original_train"

    return model, train_data, direction, test_set


def key_from_name(name):
    model, train_data, direction, test_set = infer_from_name(name)
    return model, train_data, direction, test_set


def empty_row(key):
    model, train_data, direction, test_set = key
    return {
        "model": model,
        "train_data": train_data,
        "direction": direction,
        "test_set": test_set,
        "BLEU": "pending",
        "chrF++": "pending",
        "xCOMET": "pending",
        "source_log_or_json": [],
    }


def add_source(row, source):
    source = str(source)
    if source not in row["source_log_or_json"]:
        row["source_log_or_json"].append(source)


def parse_eval_log(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    chunks = re.split(r"(?=^Direction:\s*)", text, flags=re.MULTILINE)
    records = []

    for chunk in chunks:
        if not chunk.strip() or "Final scores:" not in chunk:
            continue

        output_match = re.search(r"Predictions saved to:\s*(.+)", chunk)
        direction_match = re.search(r"^Direction:\s*(\S+)", chunk, flags=re.MULTILINE)
        bleu_match = re.search(r"^BLEU:\s*([0-9.]+)", chunk, flags=re.MULTILINE)
        chrf_match = re.search(r"^chrF\+\+:\s*([0-9.]+)", chunk, flags=re.MULTILINE)

        name_source = output_match.group(1).strip() if output_match else path.name
        key = list(key_from_name(name_source))
        if direction_match:
            key[2] = direction_match.group(1)

        records.append(
            {
                "key": tuple(key),
                "BLEU": bleu_match.group(1) if bleu_match else "pending",
                "chrF++": chrf_match.group(1) if chrf_match else "pending",
                "source": path,
            }
        )

    return records


def parse_xcomet_json(path):
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    score = data.get("system_score", "pending")
    if score != "pending" and score is not None:
        score = str(score)
    return {"key": key_from_name(path.name), "xCOMET": score, "source": path}


def main():
    rows = {}

    for log_pattern in ("eval_nllb_dedup_*.out", "eval_qwen_lora_dedup_*.out"):
        for log_name in sorted(glob.glob(str(LOG_DIR / log_pattern))):
            log_path = Path(log_name)
            for record in parse_eval_log(log_path):
                row = rows.setdefault(record["key"], empty_row(record["key"]))
                row["BLEU"] = record["BLEU"]
                row["chrF++"] = record["chrF++"]
                add_source(row, record["source"])

    for json_path in sorted(XCOMET_DIR.glob("*dedup*_xcomet.json")):
        record = parse_xcomet_json(json_path)
        row = rows.setdefault(record["key"], empty_row(record["key"]))
        row["xCOMET"] = record["xCOMET"]
        add_source(row, record["source"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = ["model", "train_data", "direction", "test_set", "BLEU", "chrF++", "xCOMET", "source_log_or_json"]
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(header) + "\n")
        for key in sorted(rows):
            row = rows[key]
            source = ";".join(row["source_log_or_json"]) or "pending"
            handle.write(
                "\t".join(
                    [
                        row["model"],
                        row["train_data"],
                        row["direction"],
                        row["test_set"],
                        row["BLEU"],
                        row["chrF++"],
                        row["xCOMET"],
                        source,
                    ]
                )
                + "\n"
            )

    print("\t".join(header))
    for key in sorted(rows):
        row = rows[key]
        source = ";".join(row["source_log_or_json"]) or "pending"
        print(
            "\t".join(
                [
                    row["model"],
                    row["train_data"],
                    row["direction"],
                    row["test_set"],
                    row["BLEU"],
                    row["chrF++"],
                    row["xCOMET"],
                    source,
                ]
            )
        )
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
