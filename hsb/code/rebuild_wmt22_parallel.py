import argparse
import csv
import gzip
from itertools import zip_longest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_DIR = ROOT / "data" / "wmt22_parallel"
DEFAULT_OUTPUT_DE = DEFAULT_INPUT_DIR / "wmt22_all_rebuilt.de"
DEFAULT_OUTPUT_HSB = DEFAULT_INPUT_DIR / "wmt22_all_rebuilt.hsb"

PARALLEL_GZ_PAIRS = [
    ("train.hsb-de.de.gz", "train.hsb-de.hsb.gz"),
    ("train2021.hsb-de.de.gz", "train2021.hsb-de.hsb.gz"),
]

TSV_GZ_FILES = [
    "HSB-DE_train.tsv.gz",
    "HSB-DE_dev.tsv.gz",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild merged WMT22 German-Upper Sorbian parallel files directly "
            "from the original UTF-8 gzip sources."
        )
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_de", type=Path, default=DEFAULT_OUTPUT_DE)
    parser.add_argument("--output_hsb", type=Path, default=DEFAULT_OUTPUT_HSB)
    return parser.parse_args()


def count_lines(path):
    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        return sum(1 for _ in handle)


def write_pair(de_out, hsb_out, de_text, hsb_text):
    de_out.write(de_text + "\n")
    hsb_out.write(hsb_text + "\n")


def append_aligned_gzip_pair(input_dir, de_name, hsb_name, de_out, hsb_out):
    de_path = input_dir / de_name
    hsb_path = input_dir / hsb_name

    kept = 0
    skipped_empty = 0
    seen = 0

    with (
        gzip.open(de_path, "rt", encoding="utf-8", errors="strict", newline="") as de_file,
        gzip.open(hsb_path, "rt", encoding="utf-8", errors="strict", newline="") as hsb_file,
    ):
        for line_no, (de_line, hsb_line) in enumerate(
            zip_longest(de_file, hsb_file, fillvalue=None), start=1
        ):
            if de_line is None or hsb_line is None:
                raise ValueError(
                    f"Input files are not line-aligned: {de_name} and {hsb_name}; "
                    f"mismatch first seen at line {line_no}"
                )

            seen += 1
            de_text = de_line.rstrip("\r\n")
            hsb_text = hsb_line.rstrip("\r\n")

            if not de_text.strip() or not hsb_text.strip():
                skipped_empty += 1
                continue

            write_pair(de_out, hsb_out, de_text, hsb_text)
            kept += 1

    print(
        f"{de_name} + {hsb_name}: kept={kept} "
        f"skipped_empty={skipped_empty} read={seen}"
    )
    return kept


def append_tsv_gzip(input_dir, tsv_name, de_out, hsb_out):
    tsv_path = input_dir / tsv_name

    kept = 0
    skipped_empty = 0
    skipped_malformed = 0
    seen = 0

    with gzip.open(
        tsv_path, "rt", encoding="utf-8", errors="strict", newline=""
    ) as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")

        for row in reader:
            seen += 1

            if len(row) < 2:
                skipped_malformed += 1
                continue

            hsb_text = row[0].strip()
            de_text = row[1].strip()

            if not de_text or not hsb_text:
                skipped_empty += 1
                continue

            write_pair(de_out, hsb_out, de_text, hsb_text)
            kept += 1

    print(
        f"{tsv_name}: kept={kept} skipped_empty={skipped_empty} "
        f"skipped_malformed={skipped_malformed} read={seen}"
    )
    return kept


def rebuild(args):
    input_dir = args.input_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    for de_name, hsb_name in PARALLEL_GZ_PAIRS:
        for filename in (de_name, hsb_name):
            path = input_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Required input file not found: {path}")

    for filename in TSV_GZ_FILES:
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    args.output_de.parent.mkdir(parents=True, exist_ok=True)
    args.output_hsb.parent.mkdir(parents=True, exist_ok=True)

    total = 0

    with (
        args.output_de.open("w", encoding="utf-8", errors="strict", newline="\n") as de_out,
        args.output_hsb.open("w", encoding="utf-8", errors="strict", newline="\n") as hsb_out,
    ):
        for de_name, hsb_name in PARALLEL_GZ_PAIRS:
            total += append_aligned_gzip_pair(
                input_dir=input_dir,
                de_name=de_name,
                hsb_name=hsb_name,
                de_out=de_out,
                hsb_out=hsb_out,
            )

        for tsv_name in TSV_GZ_FILES:
            total += append_tsv_gzip(
                input_dir=input_dir,
                tsv_name=tsv_name,
                de_out=de_out,
                hsb_out=hsb_out,
            )

    de_lines = count_lines(args.output_de)
    hsb_lines = count_lines(args.output_hsb)

    if de_lines != hsb_lines:
        raise ValueError(
            f"Output line counts do not match: {args.output_de} has {de_lines}, "
            f"{args.output_hsb} has {hsb_lines}"
        )

    if de_lines != total:
        raise ValueError(
            f"Unexpected output line count: counted {de_lines} lines but wrote {total} pairs"
        )

    print(f"final_total={total}")
    print(f"output_de={args.output_de}")
    print(f"output_hsb={args.output_hsb}")


def main():
    args = parse_args()
    rebuild(args)


if __name__ == "__main__":
    main()
