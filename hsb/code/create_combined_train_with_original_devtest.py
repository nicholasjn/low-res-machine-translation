import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_KDE4_SPLIT_DIR = ROOT / "data" / "splits"
DEFAULT_WMT22_DE = ROOT / "data" / "wmt22_parallel" / "wmt22_lid_filtered.de"
DEFAULT_WMT22_HSB = ROOT / "data" / "wmt22_parallel" / "wmt22_lid_filtered.hsb"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "splits_kde4_wmt22_lid"
DEFAULT_REPORT_PATH = (
    ROOT / "outputs" / "data_reports" / "splits_kde4_wmt22_lid_report.txt"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create KDE4+WMT22-LID train split while keeping original KDE4 "
            "dev/test splits unchanged."
        )
    )
    parser.add_argument("--kde4_split_dir", type=Path, default=DEFAULT_KDE4_SPLIT_DIR)
    parser.add_argument("--wmt22_de", type=Path, default=DEFAULT_WMT22_DE)
    parser.add_argument("--wmt22_hsb", type=Path, default=DEFAULT_WMT22_HSB)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report_path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def read_lines(path):
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\r\n") for line in handle]


def read_parallel(de_path, hsb_path):
    de_lines = read_lines(de_path)
    hsb_lines = read_lines(hsb_path)

    if len(de_lines) != len(hsb_lines):
        raise ValueError(
            f"Line count mismatch: {de_path} has {len(de_lines)}, "
            f"{hsb_path} has {len(hsb_lines)}"
        )

    return list(zip(de_lines, hsb_lines))


def write_parallel(output_dir, split_name, pairs):
    de_path = output_dir / f"{split_name}.de"
    hsb_path = output_dir / f"{split_name}.hsb"

    output_dir.mkdir(parents=True, exist_ok=True)

    with (
        de_path.open("w", encoding="utf-8", newline="\n") as de_file,
        hsb_path.open("w", encoding="utf-8", newline="\n") as hsb_file,
    ):
        for de_text, hsb_text in pairs:
            de_file.write(de_text + "\n")
            hsb_file.write(hsb_text + "\n")


def copy_split_file(src_dir, output_dir, split_name, suffix):
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_dir / f"{split_name}.{suffix}", output_dir / f"{split_name}.{suffix}")


def count_lines(path):
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def verify_split(output_dir, split_name, expected_size):
    de_path = output_dir / f"{split_name}.de"
    hsb_path = output_dir / f"{split_name}.hsb"
    de_count = count_lines(de_path)
    hsb_count = count_lines(hsb_path)

    if de_count != hsb_count:
        raise ValueError(
            f"Output line count mismatch for {split_name}: "
            f"{de_path} has {de_count}, {hsb_path} has {hsb_count}"
        )

    if de_count != expected_size:
        raise ValueError(
            f"Unexpected {split_name} size: expected {expected_size}, got {de_count}"
        )


def filter_combined_train(kde4_train_pairs, wmt22_pairs, heldout_pairs):
    combined_pairs = kde4_train_pairs + wmt22_pairs
    seen_train = set()
    final_train_pairs = []
    duplicate_removed = 0
    leakage_removed = 0

    for pair in combined_pairs:
        if pair in seen_train:
            duplicate_removed += 1
            continue

        seen_train.add(pair)

        if pair in heldout_pairs:
            leakage_removed += 1
            continue

        final_train_pairs.append(pair)

    return final_train_pairs, duplicate_removed, leakage_removed, len(combined_pairs)


def write_report(report_path, lines):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_splits(args):
    kde4_train_pairs = read_parallel(
        args.kde4_split_dir / "train.de", args.kde4_split_dir / "train.hsb"
    )
    kde4_dev_pairs = read_parallel(
        args.kde4_split_dir / "dev.de", args.kde4_split_dir / "dev.hsb"
    )
    kde4_test_pairs = read_parallel(
        args.kde4_split_dir / "test.de", args.kde4_split_dir / "test.hsb"
    )
    wmt22_pairs = read_parallel(args.wmt22_de, args.wmt22_hsb)

    heldout_pairs = set(kde4_dev_pairs) | set(kde4_test_pairs)
    final_train_pairs, duplicate_removed, leakage_removed, combined_before_dedup = (
        filter_combined_train(
            kde4_train_pairs=kde4_train_pairs,
            wmt22_pairs=wmt22_pairs,
            heldout_pairs=heldout_pairs,
        )
    )

    write_parallel(args.output_dir, "train", final_train_pairs)

    for split_name in ("dev", "test"):
        for suffix in ("de", "hsb"):
            copy_split_file(args.kde4_split_dir, args.output_dir, split_name, suffix)

    verify_split(args.output_dir, "train", len(final_train_pairs))
    verify_split(args.output_dir, "dev", len(kde4_dev_pairs))
    verify_split(args.output_dir, "test", len(kde4_test_pairs))

    report_lines = [
        f"original_kde4_train_size\t{len(kde4_train_pairs)}",
        f"original_kde4_dev_size\t{len(kde4_dev_pairs)}",
        f"original_kde4_test_size\t{len(kde4_test_pairs)}",
        f"wmt22_filtered_size\t{len(wmt22_pairs)}",
        f"combined_train_before_dedup\t{combined_before_dedup}",
        f"duplicate_train_pairs_removed\t{duplicate_removed}",
        f"dev_test_leakage_pairs_removed\t{leakage_removed}",
        f"final_train_size\t{len(final_train_pairs)}",
        f"final_dev_size\t{len(kde4_dev_pairs)}",
        f"final_test_size\t{len(kde4_test_pairs)}",
        f"output_dir\t{args.output_dir}",
    ]
    write_report(args.report_path, report_lines)

    for line in report_lines:
        print(line)
    print(f"report_path\t{args.report_path}")


def main():
    args = parse_args()
    create_splits(args)


if __name__ == "__main__":
    main()
