import argparse
import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LID_TSV = ROOT / "outputs" / "lid_reports" / "wmt22_rebuilt_lid_sample.tsv"
DEFAULT_SUMMARY_TSV = ROOT / "outputs" / "lid_reports" / "wmt22_rebuilt_lid_summary.tsv"
DEFAULT_CATEGORIZED_TSV = (
    ROOT / "outputs" / "lid_reports" / "wmt22_rebuilt_lid_categorized.tsv"
)

EXPECTED_DE_LABEL = "__label__deu_Latn"
EXPECTED_HSB_LABEL = "__label__hsb_Latn"

RELATED_SLAVIC_LABELS = {
    "__label__dsb_Latn",
    "__label__ces_Latn",
    "__label__pol_Latn",
    "__label__slk_Latn",
    "__label__slv_Latn",
    "__label__hrv_Latn",
    "__label__bos_Latn",
    "__label__srp_Latn",
}

REQUIRED_COLUMNS = [
    "de_pred1_label",
    "de_pred2_label",
    "de_pred3_label",
    "hsb_pred1_label",
    "hsb_pred2_label",
    "hsb_pred3_label",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize and categorize a GlotLID TSV report."
    )
    parser.add_argument("--lid_tsv", type=Path, default=DEFAULT_LID_TSV)
    parser.add_argument("--summary_tsv", type=Path, default=DEFAULT_SUMMARY_TSV)
    parser.add_argument(
        "--categorized_tsv", type=Path, default=DEFAULT_CATEGORIZED_TSV
    )
    return parser.parse_args()


def labels_top3(row, prefix):
    return [
        row[f"{prefix}_pred1_label"],
        row[f"{prefix}_pred2_label"],
        row[f"{prefix}_pred3_label"],
    ]


def categorize(row):
    de_top3 = labels_top3(row, "de")
    hsb_top3 = labels_top3(row, "hsb")

    strict_pass = (
        row["de_pred1_label"] == EXPECTED_DE_LABEL
        and row["hsb_pred1_label"] == EXPECTED_HSB_LABEL
    )

    relaxed_pass = (
        EXPECTED_DE_LABEL in de_top3
        and (
            EXPECTED_HSB_LABEL in hsb_top3
            or row["hsb_pred1_label"] in RELATED_SLAVIC_LABELS
        )
    )

    if strict_pass:
        return "strict_pass"

    if relaxed_pass:
        return "relaxed_pass"

    return "review"


def validate_columns(fieldnames):
    if fieldnames is None:
        raise ValueError("Input TSV is empty or missing a header row")

    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(f"Input TSV is missing required columns: {', '.join(missing)}")


def write_summary(summary_tsv, category_counts, hsb_pred1_counts, total):
    summary_tsv.parent.mkdir(parents=True, exist_ok=True)

    with summary_tsv.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.writer(summary_file, delimiter="\t", lineterminator="\n")
        writer.writerow(["section", "label", "count"])

        writer.writerow(["category", "strict_pass", category_counts["strict_pass"]])
        writer.writerow(["category", "relaxed_pass", category_counts["relaxed_pass"]])
        writer.writerow(["category", "review", category_counts["review"]])
        writer.writerow(["category", "total", total])

        for label, count in hsb_pred1_counts.most_common():
            writer.writerow(["hsb_pred1_label", label, count])


def print_summary(category_counts, hsb_pred1_counts, total, summary_tsv, categorized_tsv):
    print("category\tcount")
    print(f"strict_pass\t{category_counts['strict_pass']}")
    print(f"relaxed_pass\t{category_counts['relaxed_pass']}")
    print(f"review\t{category_counts['review']}")
    print(f"total\t{total}")
    print()
    print("hsb_pred1_label\tcount")

    for label, count in hsb_pred1_counts.most_common():
        print(f"{label}\t{count}")

    print()
    print(f"summary_tsv\t{summary_tsv}")
    print(f"categorized_tsv\t{categorized_tsv}")


def summarize(args):
    if not args.lid_tsv.exists():
        raise FileNotFoundError(f"LID TSV not found: {args.lid_tsv}")

    category_counts = Counter()
    hsb_pred1_counts = Counter()
    total = 0

    args.categorized_tsv.parent.mkdir(parents=True, exist_ok=True)

    with (
        args.lid_tsv.open("r", encoding="utf-8", newline="") as lid_file,
        args.categorized_tsv.open("w", encoding="utf-8", newline="") as categorized_file,
    ):
        reader = csv.DictReader(lid_file, delimiter="\t")
        validate_columns(reader.fieldnames)

        fieldnames = ["category"] + reader.fieldnames
        writer = csv.DictWriter(
            categorized_file,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()

        for row in reader:
            category = categorize(row)
            total += 1
            category_counts[category] += 1
            hsb_pred1_counts[row["hsb_pred1_label"]] += 1

            categorized_row = {"category": category}
            categorized_row.update(row)
            writer.writerow(categorized_row)

    write_summary(
        summary_tsv=args.summary_tsv,
        category_counts=category_counts,
        hsb_pred1_counts=hsb_pred1_counts,
        total=total,
    )
    print_summary(
        category_counts=category_counts,
        hsb_pred1_counts=hsb_pred1_counts,
        total=total,
        summary_tsv=args.summary_tsv,
        categorized_tsv=args.categorized_tsv,
    )


def main():
    args = parse_args()
    summarize(args)


if __name__ == "__main__":
    main()
