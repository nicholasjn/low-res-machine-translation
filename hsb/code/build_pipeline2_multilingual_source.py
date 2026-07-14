import argparse
import csv
import gzip
from collections import Counter
from itertools import zip_longest
from pathlib import Path


LANG_CONFIGS = [
    ("cs", "ces_Latn", "cs_path", "cs_hsb_path"),
    ("en", "eng_Latn", "en_path", "en_hsb_path"),
    ("pl", "pol_Latn", "pl_path", "pl_hsb_path"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a multilingual non-German to Upper Sorbian source dataset for Pipeline 2."
    )
    parser.add_argument("--cs_path", required=True, type=Path)
    parser.add_argument("--cs_hsb_path", required=True, type=Path)
    parser.add_argument("--en_path", required=True, type=Path)
    parser.add_argument("--en_hsb_path", required=True, type=Path)
    parser.add_argument("--pl_path", required=True, type=Path)
    parser.add_argument("--pl_hsb_path", required=True, type=Path)
    parser.add_argument("--output_tsv", required=True, type=Path)
    parser.add_argument("--output_src", required=True, type=Path)
    parser.add_argument("--output_hsb", required=True, type=Path)
    parser.add_argument("--output_lang", required=True, type=Path)
    parser.add_argument("--max_pairs", type=int, default=-1)
    return parser.parse_args()


def open_text(path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", errors="replace", newline="")
    return path.open("r", encoding="utf-8-sig", errors="replace", newline="")


def read_pair(src_path, hsb_path, short_lang, nllb_lang):
    rows = []
    src_count = 0
    hsb_count = 0

    with open_text(src_path) as src_file, open_text(hsb_path) as hsb_file:
        for src_line, hsb_line in zip_longest(src_file, hsb_file):
            if src_line is not None:
                src_count += 1
            if hsb_line is not None:
                hsb_count += 1
            if src_line is None or hsb_line is None:
                continue

            src_sentence = src_line.rstrip("\n\r")
            hsb_sentence = hsb_line.rstrip("\n\r")
            if not src_sentence.strip() or not hsb_sentence.strip():
                continue
            rows.append((short_lang, nllb_lang, src_sentence, hsb_sentence))

    if src_count != hsb_count:
        raise SystemExit(
            f"Line count mismatch for {short_lang}: "
            f"{src_path} has {src_count}, {hsb_path} has {hsb_count}"
        )

    return rows, src_count


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def write_outputs(rows, output_tsv, output_src, output_hsb, output_lang):
    for path in (output_tsv, output_src, output_hsb, output_lang):
        ensure_parent(path)

    with output_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["src_lang", "src_sentence", "hsb_sentence"])
        for src_lang, _, src_sentence, hsb_sentence in rows:
            writer.writerow([src_lang, src_sentence, hsb_sentence])

    with output_src.open("w", encoding="utf-8", newline="\n") as src_file, output_hsb.open(
        "w", encoding="utf-8", newline="\n"
    ) as hsb_file, output_lang.open("w", encoding="utf-8", newline="\n") as lang_file:
        for _, nllb_lang, src_sentence, hsb_sentence in rows:
            src_file.write(src_sentence + "\n")
            hsb_file.write(hsb_sentence + "\n")
            lang_file.write(nllb_lang + "\n")


def main():
    args = parse_args()

    seen = set()
    merged_rows = []
    kept_before_truncation = Counter()
    duplicate_count = 0

    for short_lang, nllb_lang, src_attr, hsb_attr in LANG_CONFIGS:
        src_path = getattr(args, src_attr)
        hsb_path = getattr(args, hsb_attr)
        rows, raw_count = read_pair(src_path, hsb_path, short_lang, nllb_lang)

        for row in rows:
            dedupe_key = (row[0], row[2], row[3])
            if dedupe_key in seen:
                duplicate_count += 1
                continue
            seen.add(dedupe_key)
            merged_rows.append(row)
            kept_before_truncation[nllb_lang] += 1

        skipped = raw_count - len(rows)
        print(f"{short_lang}_raw_lines\t{raw_count}")
        print(f"{short_lang}_nonempty_pairs\t{len(rows)}")
        print(f"{short_lang}_empty_or_unaligned_skipped\t{skipped}")

    if args.max_pairs > 0:
        merged_rows = merged_rows[: args.max_pairs]

    final_counts = Counter(nllb_lang for _, nllb_lang, _, _ in merged_rows)
    write_outputs(merged_rows, args.output_tsv, args.output_src, args.output_hsb, args.output_lang)

    print("counts_per_language")
    for _, nllb_lang, _, _ in LANG_CONFIGS:
        print(f"{nllb_lang}\t{final_counts[nllb_lang]}")
    print(f"total\t{len(merged_rows)}")
    print(f"unique_before_truncation\t{sum(kept_before_truncation.values())}")
    print(f"duplicates_removed\t{duplicate_count}")
    print(f"output_tsv\t{args.output_tsv}")
    print(f"output_src\t{args.output_src}")
    print(f"output_hsb\t{args.output_hsb}")
    print(f"output_lang\t{args.output_lang}")


if __name__ == "__main__":
    main()
