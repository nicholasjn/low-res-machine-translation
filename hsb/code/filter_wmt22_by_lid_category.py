import argparse
import csv
from itertools import zip_longest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DE_PATH = ROOT / "data" / "wmt22_parallel" / "wmt22_all_rebuilt.de"
DEFAULT_HSB_PATH = ROOT / "data" / "wmt22_parallel" / "wmt22_all_rebuilt.hsb"
DEFAULT_CATEGORIZED_TSV = (
    ROOT / "outputs" / "lid_reports" / "wmt22_rebuilt_lid_categorized.tsv"
)
DEFAULT_OUTPUT_DE = ROOT / "data" / "wmt22_parallel" / "wmt22_lid_filtered.de"
DEFAULT_OUTPUT_HSB = ROOT / "data" / "wmt22_parallel" / "wmt22_lid_filtered.hsb"
DEFAULT_REMOVED_TSV = (
    ROOT / "outputs" / "lid_reports" / "wmt22_lid_filtered_removed.tsv"
)

KEEP_CATEGORIES = {"strict_pass", "relaxed_pass"}
REMOVE_CATEGORY = "review"
VALID_CATEGORIES = KEEP_CATEGORIES | {REMOVE_CATEGORY}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter rebuilt WMT22 parallel files by categorized GlotLID output."
    )
    parser.add_argument("--de_path", type=Path, default=DEFAULT_DE_PATH)
    parser.add_argument("--hsb_path", type=Path, default=DEFAULT_HSB_PATH)
    parser.add_argument("--categorized_tsv", type=Path, default=DEFAULT_CATEGORIZED_TSV)
    parser.add_argument("--output_de", type=Path, default=DEFAULT_OUTPUT_DE)
    parser.add_argument("--output_hsb", type=Path, default=DEFAULT_OUTPUT_HSB)
    parser.add_argument("--removed_tsv", type=Path, default=DEFAULT_REMOVED_TSV)
    return parser.parse_args()


def require_file(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def validate_fieldnames(fieldnames):
    if fieldnames is None:
        raise ValueError("Categorized TSV is empty or missing a header row")

    if "category" not in fieldnames:
        raise ValueError("Categorized TSV is missing required column: category")


def filter_pairs(args):
    require_file(args.de_path, "German input file")
    require_file(args.hsb_path, "Upper Sorbian input file")
    require_file(args.categorized_tsv, "Categorized LID TSV")

    args.output_de.parent.mkdir(parents=True, exist_ok=True)
    args.output_hsb.parent.mkdir(parents=True, exist_ok=True)
    args.removed_tsv.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    removed = 0

    with (
        args.de_path.open("r", encoding="utf-8", newline="") as de_file,
        args.hsb_path.open("r", encoding="utf-8", newline="") as hsb_file,
        args.categorized_tsv.open("r", encoding="utf-8-sig", newline="") as category_file,
        args.output_de.open("w", encoding="utf-8", newline="\n") as de_out,
        args.output_hsb.open("w", encoding="utf-8", newline="\n") as hsb_out,
        args.removed_tsv.open("w", encoding="utf-8", newline="") as removed_file,
    ):
        reader = csv.DictReader(category_file, delimiter="\t")
        validate_fieldnames(reader.fieldnames)

        removed_fieldnames = [
            "source_line_no",
            "source_de_text",
            "source_hsb_text",
        ] + reader.fieldnames
        removed_writer = csv.DictWriter(
            removed_file,
            fieldnames=removed_fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        removed_writer.writeheader()

        aligned_rows = zip_longest(de_file, hsb_file, reader, fillvalue=None)

        for line_no, (de_line, hsb_line, category_row) in enumerate(
            aligned_rows, start=1
        ):
            if de_line is None or hsb_line is None or category_row is None:
                raise ValueError(
                    "Input line counts do not match: "
                    f"mismatch first seen at row {line_no}"
                )

            total += 1
            category = category_row["category"]

            if category not in VALID_CATEGORIES:
                raise ValueError(
                    f"Unexpected category at row {line_no}: {category!r}. "
                    f"Expected one of: {', '.join(sorted(VALID_CATEGORIES))}"
                )

            de_text = de_line.rstrip("\r\n")
            hsb_text = hsb_line.rstrip("\r\n")

            if category in KEEP_CATEGORIES:
                de_out.write(de_text + "\n")
                hsb_out.write(hsb_text + "\n")
                kept += 1
                continue

            removed_row = {
                "source_line_no": line_no,
                "source_de_text": de_text,
                "source_hsb_text": hsb_text,
            }
            removed_row.update(category_row)
            removed_writer.writerow(removed_row)
            removed += 1

    print(f"total_input\t{total}")
    print(f"kept\t{kept}")
    print(f"removed\t{removed}")
    print(f"output_de\t{args.output_de}")
    print(f"output_hsb\t{args.output_hsb}")
    print(f"removed_tsv\t{args.removed_tsv}")


def main():
    args = parse_args()
    filter_pairs(args)


if __name__ == "__main__":
    main()
