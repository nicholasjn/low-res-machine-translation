from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "data_leakage_check"


def normalize(text):
    return text.strip()


def read_lines(path):
    with path.open("r", encoding="utf-8") as handle:
        return [normalize(line.rstrip("\n")) for line in handle]


def load_split(split_dir, split_name):
    de_path = split_dir / f"{split_name}.de"
    hsb_path = split_dir / f"{split_name}.hsb"
    de_lines = read_lines(de_path)
    hsb_lines = read_lines(hsb_path)
    n = min(len(de_lines), len(hsb_lines))
    if len(de_lines) != len(hsb_lines):
        print(
            f"WARNING: {de_path} has {len(de_lines)} lines but "
            f"{hsb_path} has {len(hsb_lines)} lines; using first {n} pairs."
        )
    pairs = list(zip(de_lines[:n], hsb_lines[:n]))
    return {
        "de": de_lines[:n],
        "hsb": hsb_lines[:n],
        "pairs": pairs,
        "pair_set": set(pairs),
        "de_set": set(de_lines[:n]),
        "hsb_set": set(hsb_lines[:n]),
    }


def safe_name(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def write_tsv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(str(cell).replace("\t", " ") for cell in row) + "\n")


def pair_check(name, left_label, left_pairs, right_label, right_pairs):
    left_set = set(left_pairs)
    right_set = set(right_pairs)
    overlap = sorted(left_set & right_set)
    smaller = min(len(left_set), len(right_set))
    percent = (len(overlap) / smaller * 100.0) if smaller else 0.0
    return {
        "check_name": name,
        "check_type": "pair",
        "left": left_label,
        "right": right_label,
        "left_n": len(left_set),
        "right_n": len(right_set),
        "overlap_n": len(overlap),
        "percent_of_smaller": percent,
        "examples": overlap[:50],
    }


def single_side_check(name, check_type, left_label, left_items, right_label, right_items):
    left_set = set(left_items)
    right_set = set(right_items)
    overlap = sorted(left_set & right_set)
    smaller = min(len(left_set), len(right_set))
    percent = (len(overlap) / smaller * 100.0) if smaller else 0.0
    return {
        "check_name": name,
        "check_type": check_type,
        "left": left_label,
        "right": right_label,
        "left_n": len(left_set),
        "right_n": len(right_set),
        "overlap_n": len(overlap),
        "percent_of_smaller": percent,
        "examples": overlap[:50],
    }


def save_examples(result):
    if result["overlap_n"] == 0:
        return

    path = OUTPUT_DIR / f"examples_{safe_name(result['check_name'])}.tsv"
    if result["check_type"] == "pair":
        write_tsv(path, ["de", "hsb"], result["examples"])
    elif result["check_type"] == "source":
        write_tsv(path, ["de"], [(item,) for item in result["examples"]])
    else:
        write_tsv(path, ["hsb"], [(item,) for item in result["examples"]])


def interpretation(results):
    lines = ["", "Automatic interpretation:"]

    for split_prefix, label in (
        ("original", "original"),
        ("cleaned", "cleaned"),
    ):
        train_test = next(
            result
            for result in results
            if result["check_name"] == f"{split_prefix}_train_vs_test_pair_overlap"
        )
        if train_test["overlap_n"] == 0:
            lines.append(
                f"- No exact parallel sentence-pair leakage was found for this split: {label}."
            )
        else:
            lines.append(
                f"- Exact train-test parallel sentence-pair leakage was found for this split: "
                f"{label} ({train_test['overlap_n']} pairs)."
            )

    side_overlap = [
        result
        for result in results
        if result["check_type"] in {"source", "target"} and result["overlap_n"] > 0
    ]
    if side_overlap:
        lines.append(
            "- Source-only or target-only overlap is non-zero. This may be expected for "
            "KDE4 UI data because of short repeated interface strings, and is weaker "
            "evidence than exact pair overlap."
        )

    cleaned_train_original_test = next(
        result
        for result in results
        if result["check_name"] == "cleaned_train_vs_original_test_pair_overlap"
    )
    if cleaned_train_original_test["overlap_n"] > 0:
        lines.append(
            "- IMPORTANT: cleaned train vs original test has non-zero exact pair overlap. "
            "This matters because cleaned-trained models are evaluated on the original test set."
        )

    return lines


def main():
    original_dir = ROOT / "data" / "splits"
    cleaned_dir = ROOT / "data" / "splits_cleaned"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original = {
        split: load_split(original_dir, split)
        for split in ("train", "dev", "test")
    }
    cleaned = {
        split: load_split(cleaned_dir, split)
        for split in ("train", "dev", "test")
    }

    results = [
        pair_check(
            "original_train_vs_dev_pair_overlap",
            "original/train",
            original["train"]["pairs"],
            "original/dev",
            original["dev"]["pairs"],
        ),
        pair_check(
            "original_train_vs_test_pair_overlap",
            "original/train",
            original["train"]["pairs"],
            "original/test",
            original["test"]["pairs"],
        ),
        pair_check(
            "original_dev_vs_test_pair_overlap",
            "original/dev",
            original["dev"]["pairs"],
            "original/test",
            original["test"]["pairs"],
        ),
        single_side_check(
            "original_train_de_vs_test_de_source_overlap",
            "source",
            "original/train.de",
            original["train"]["de"],
            "original/test.de",
            original["test"]["de"],
        ),
        single_side_check(
            "original_train_hsb_vs_test_hsb_target_overlap",
            "target",
            "original/train.hsb",
            original["train"]["hsb"],
            "original/test.hsb",
            original["test"]["hsb"],
        ),
        pair_check(
            "cleaned_train_vs_dev_pair_overlap",
            "cleaned/train",
            cleaned["train"]["pairs"],
            "cleaned/dev",
            cleaned["dev"]["pairs"],
        ),
        pair_check(
            "cleaned_train_vs_test_pair_overlap",
            "cleaned/train",
            cleaned["train"]["pairs"],
            "cleaned/test",
            cleaned["test"]["pairs"],
        ),
        pair_check(
            "cleaned_dev_vs_test_pair_overlap",
            "cleaned/dev",
            cleaned["dev"]["pairs"],
            "cleaned/test",
            cleaned["test"]["pairs"],
        ),
        single_side_check(
            "cleaned_train_de_vs_test_de_source_overlap",
            "source",
            "cleaned/train.de",
            cleaned["train"]["de"],
            "cleaned/test.de",
            cleaned["test"]["de"],
        ),
        single_side_check(
            "cleaned_train_hsb_vs_test_hsb_target_overlap",
            "target",
            "cleaned/train.hsb",
            cleaned["train"]["hsb"],
            "cleaned/test.hsb",
            cleaned["test"]["hsb"],
        ),
        pair_check(
            "original_train_vs_cleaned_test_pair_overlap",
            "original/train",
            original["train"]["pairs"],
            "cleaned/test",
            cleaned["test"]["pairs"],
        ),
        pair_check(
            "cleaned_train_vs_original_test_pair_overlap",
            "cleaned/train",
            cleaned["train"]["pairs"],
            "original/test",
            original["test"]["pairs"],
        ),
        pair_check(
            "original_test_vs_cleaned_test_pair_overlap",
            "original/test",
            original["test"]["pairs"],
            "cleaned/test",
            cleaned["test"]["pairs"],
        ),
        pair_check(
            "original_train_vs_cleaned_train_pair_overlap",
            "original/train",
            original["train"]["pairs"],
            "cleaned/train",
            cleaned["train"]["pairs"],
        ),
    ]

    for result in results:
        save_examples(result)

    summary_rows = [
        (
            result["check_name"],
            result["check_type"],
            result["left"],
            result["right"],
            result["left_n"],
            result["right_n"],
            result["overlap_n"],
            f"{result['percent_of_smaller']:.4f}",
        )
        for result in results
    ]

    write_tsv(
        OUTPUT_DIR / "leakage_summary.tsv",
        [
            "check_name",
            "check_type",
            "left",
            "right",
            "left_unique_n",
            "right_unique_n",
            "overlap_n",
            "percent_of_smaller",
        ],
        summary_rows,
    )

    text_lines = [
        "Data leakage check",
        f"Output directory: {OUTPUT_DIR}",
        "",
        "check_name\ttype\toverlap\tpercent_of_smaller",
    ]
    for result in results:
        text_lines.append(
            f"{result['check_name']}\t{result['check_type']}\t"
            f"{result['overlap_n']}\t{result['percent_of_smaller']:.2f}%"
        )
    text_lines.extend(interpretation(results))

    summary_text = "\n".join(text_lines)
    (OUTPUT_DIR / "leakage_summary.txt").write_text(summary_text + "\n", encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
