from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DIR = ROOT / "data" / "splits"
CLEANED_DIR = ROOT / "data" / "splits_cleaned"
OUTPUT_DIR = ROOT / "outputs" / "testset_diff"


def read_lines(path):
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def load_pairs(de_path, hsb_path):
    de_lines = read_lines(de_path)
    hsb_lines = read_lines(hsb_path)
    n = min(len(de_lines), len(hsb_lines))
    if len(de_lines) != len(hsb_lines):
        print(
            f"WARNING: {de_path} has {len(de_lines)} lines but "
            f"{hsb_path} has {len(hsb_lines)} lines; using first {n} pairs."
        )
    return list(zip(de_lines[:n], hsb_lines[:n]))


def token_count(text):
    return len(text.split())


def likely_reason(de_sentence, hsb_sentence):
    if de_sentence.strip() == hsb_sentence.strip():
        return "identical"

    de_tokens = token_count(de_sentence)
    hsb_tokens = token_count(hsb_sentence)
    if de_tokens > 200 or hsb_tokens > 200:
        return "too_long"

    if de_tokens == 0 or hsb_tokens == 0:
        if de_tokens != hsb_tokens:
            return "length_ratio"
        return "other_or_resplit"

    ratio = max(de_tokens / hsb_tokens, hsb_tokens / de_tokens)
    if ratio > 3.0:
        return "length_ratio"

    return "other_or_resplit"


def write_tsv(path, rows, header):
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(str(cell).replace("\t", " ") for cell in row) + "\n")


def counter_to_ordered_rows(source_pairs, pair_counts):
    remaining = Counter(pair_counts)
    rows = []
    for pair in source_pairs:
        if remaining[pair] <= 0:
            continue
        rows.append(pair)
        remaining[pair] -= 1
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original_pairs = load_pairs(ORIGINAL_DIR / "test.de", ORIGINAL_DIR / "test.hsb")
    cleaned_pairs = load_pairs(CLEANED_DIR / "test.de", CLEANED_DIR / "test.hsb")

    original_counts = Counter(original_pairs)
    cleaned_counts = Counter(cleaned_pairs)

    common_counts = original_counts & cleaned_counts
    original_only_counts = original_counts - cleaned_counts
    cleaned_only_counts = cleaned_counts - original_counts

    common_pairs = counter_to_ordered_rows(original_pairs, common_counts)
    original_only = []
    examples_by_reason = defaultdict(list)
    remaining_original_only = Counter(original_only_counts)
    for pair in original_pairs:
        if remaining_original_only[pair] <= 0:
            continue
        remaining_original_only[pair] -= 1
        de_sentence, hsb_sentence = pair
        reason = likely_reason(de_sentence, hsb_sentence)
        original_only.append((de_sentence, hsb_sentence, reason))
        if len(examples_by_reason[reason]) < 30:
            examples_by_reason[reason].append((de_sentence, hsb_sentence, reason))

    cleaned_only = counter_to_ordered_rows(cleaned_pairs, cleaned_only_counts)

    common_count = sum(common_counts.values())
    original_only_count = sum(original_only_counts.values())
    cleaned_only_count = sum(cleaned_only_counts.values())
    overlap_pct = (common_count / len(original_pairs) * 100.0) if original_pairs else 0.0

    write_tsv(OUTPUT_DIR / "common_test_pairs.tsv", common_pairs, ["de", "hsb"])
    write_tsv(
        OUTPUT_DIR / "original_test_only.tsv",
        original_only,
        ["de", "hsb", "likely_reason"],
    )
    write_tsv(OUTPUT_DIR / "cleaned_test_only.tsv", cleaned_only, ["de", "hsb"])

    for reason in ("identical", "too_long", "length_ratio", "other_or_resplit"):
        write_tsv(
            OUTPUT_DIR / f"examples_{reason}.tsv",
            examples_by_reason.get(reason, []),
            ["de", "hsb", "likely_reason"],
        )

    summary_lines = [
        "Original vs cleaned test-set comparison",
        f"Original test pairs: {len(original_pairs)}",
        f"Cleaned test pairs: {len(cleaned_pairs)}",
        f"Exact common pairs: {common_count}",
        f"Pairs only in original test: {original_only_count}",
        f"Pairs only in cleaned test: {cleaned_only_count}",
        f"Percentage overlap: {overlap_pct:.2f}% of original test pairs",
        "",
        "Caveat: the cleaned data may have been re-split after cleaning, so pairs only in",
        "the original test set were not necessarily removed by cleaning; some may simply",
        "have moved out of the cleaned test split.",
        "",
        f"Outputs written to: {OUTPUT_DIR}",
    ]
    summary = "\n".join(summary_lines)
    (OUTPUT_DIR / "testset_diff_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
