from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SPLIT_DIR = ROOT / "data" / "splits"
CLEANED_SPLIT_DIR = ROOT / "data" / "splits_cleaned"
OUT_ORIGINAL_TRAIN = ROOT / "data" / "splits_original_test_dedup_vs_original_train"
OUT_CLEANED_TRAIN = ROOT / "data" / "splits_original_test_dedup_vs_cleaned_train"
SUMMARY_DIR = ROOT / "outputs" / "data_leakage_check"


def normalize(text):
    return text.strip()


def read_lines(path):
    with path.open("r", encoding="utf-8") as handle:
        return [normalize(line.rstrip("\n")) for line in handle]


def load_pairs(split_dir, split_name):
    de_lines = read_lines(split_dir / f"{split_name}.de")
    hsb_lines = read_lines(split_dir / f"{split_name}.hsb")
    n = min(len(de_lines), len(hsb_lines))
    if len(de_lines) != len(hsb_lines):
        print(
            f"WARNING: {split_dir / (split_name + '.de')} has {len(de_lines)} lines but "
            f"{split_dir / (split_name + '.hsb')} has {len(hsb_lines)} lines; using {n} pairs."
        )
    return list(zip(de_lines[:n], hsb_lines[:n]))


def write_split(output_dir, pairs):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "test.de").open("w", encoding="utf-8", newline="") as de_handle:
        with (output_dir / "test.hsb").open("w", encoding="utf-8", newline="") as hsb_handle:
            for de_sentence, hsb_sentence in pairs:
                de_handle.write(de_sentence + "\n")
                hsb_handle.write(hsb_sentence + "\n")


def write_removed(path, removed_pairs):
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("de\thsb\n")
        for de_sentence, hsb_sentence in removed_pairs:
            handle.write(
                f"{de_sentence.replace(chr(9), ' ')}\t{hsb_sentence.replace(chr(9), ' ')}\n"
            )


def deduplicate(test_pairs, train_pairs):
    train_set = set(train_pairs)
    kept = []
    removed = []
    for pair in test_pairs:
        if pair in train_set:
            removed.append(pair)
        else:
            kept.append(pair)
    return kept, removed


def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    original_train = load_pairs(ORIGINAL_SPLIT_DIR, "train")
    original_test = load_pairs(ORIGINAL_SPLIT_DIR, "test")
    cleaned_train = load_pairs(CLEANED_SPLIT_DIR, "train")

    kept_vs_original, removed_vs_original = deduplicate(original_test, original_train)
    kept_vs_cleaned, removed_vs_cleaned = deduplicate(original_test, cleaned_train)

    write_split(OUT_ORIGINAL_TRAIN, kept_vs_original)
    write_split(OUT_CLEANED_TRAIN, kept_vs_cleaned)

    write_removed(SUMMARY_DIR / "dedup_removed_vs_original_train.tsv", removed_vs_original)
    write_removed(SUMMARY_DIR / "dedup_removed_vs_cleaned_train.tsv", removed_vs_cleaned)

    lines = [
        "Deduplicated original test-set creation",
        f"Original test pairs: {len(original_test)}",
        "",
        "Dedup vs original train:",
        f"Removed pairs: {len(removed_vs_original)}",
        f"Remaining pairs: {len(kept_vs_original)}",
        f"Output: {OUT_ORIGINAL_TRAIN}",
        "",
        "Dedup vs cleaned train:",
        f"Removed pairs: {len(removed_vs_cleaned)}",
        f"Remaining pairs: {len(kept_vs_cleaned)}",
        f"Output: {OUT_CLEANED_TRAIN}",
        "",
        f"Removed-pair TSVs: {SUMMARY_DIR}",
    ]
    summary = "\n".join(lines)
    (SUMMARY_DIR / "dedup_test_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
