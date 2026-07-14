import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_KDE4_SPLIT_DIR = ROOT / "data" / "splits"
DEFAULT_WMT22_DE = ROOT / "data" / "wmt22_parallel" / "wmt22_lid_filtered.de"
DEFAULT_WMT22_HSB = ROOT / "data" / "wmt22_parallel" / "wmt22_lid_filtered.hsb"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "splits_kde4_wmt22_lid_100k"
DEFAULT_SELECTED_WMT22_DE = (
    ROOT / "data" / "wmt22_parallel" / "wmt22_lid_filtered_selected_100k.de"
)
DEFAULT_SELECTED_WMT22_HSB = (
    ROOT / "data" / "wmt22_parallel" / "wmt22_lid_filtered_selected_100k.hsb"
)
DEFAULT_REPORT_PATH = (
    ROOT / "outputs" / "data_reports" / "splits_kde4_wmt22_lid_100k_report.txt"
)
DEFAULT_NUM_WMT22_PAIRS = 100_000


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create KDE4 train plus exactly 100,000 valid WMT22 LID-filtered "
            "pairs, keeping original KDE4 dev/test unchanged."
        )
    )
    parser.add_argument("--kde4_split_dir", type=Path, default=DEFAULT_KDE4_SPLIT_DIR)
    parser.add_argument("--wmt22_de", type=Path, default=DEFAULT_WMT22_DE)
    parser.add_argument("--wmt22_hsb", type=Path, default=DEFAULT_WMT22_HSB)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selected_wmt22_de", type=Path, default=DEFAULT_SELECTED_WMT22_DE)
    parser.add_argument(
        "--selected_wmt22_hsb", type=Path, default=DEFAULT_SELECTED_WMT22_HSB
    )
    parser.add_argument("--report_path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--num_wmt22_pairs", type=int, default=DEFAULT_NUM_WMT22_PAIRS)
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


def write_pairs(de_path, hsb_path, pairs):
    de_path.parent.mkdir(parents=True, exist_ok=True)
    hsb_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        de_path.open("w", encoding="utf-8", newline="\n") as de_file,
        hsb_path.open("w", encoding="utf-8", newline="\n") as hsb_file,
    ):
        for de_text, hsb_text in pairs:
            de_file.write(de_text + "\n")
            hsb_file.write(hsb_text + "\n")


def write_parallel_split(output_dir, split_name, pairs):
    write_pairs(
        output_dir / f"{split_name}.de",
        output_dir / f"{split_name}.hsb",
        pairs,
    )


def copy_split_file(src_dir, output_dir, split_name, suffix):
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_dir / f"{split_name}.{suffix}", output_dir / f"{split_name}.{suffix}")


def count_lines(path):
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def verify_pair_files(de_path, hsb_path, expected_size):
    de_count = count_lines(de_path)
    hsb_count = count_lines(hsb_path)

    if de_count != hsb_count:
        raise ValueError(
            f"Output line count mismatch: {de_path} has {de_count}, "
            f"{hsb_path} has {hsb_count}"
        )

    if de_count != expected_size:
        raise ValueError(f"Unexpected output size: expected {expected_size}, got {de_count}")


def verify_split(output_dir, split_name, expected_size):
    verify_pair_files(
        output_dir / f"{split_name}.de",
        output_dir / f"{split_name}.hsb",
        expected_size,
    )


def select_wmt22_pairs(wmt22_pairs, kde4_pairs, requested_count):
    if requested_count < 1:
        raise ValueError("--num_wmt22_pairs must be a positive integer")

    selected = []
    seen_wmt22 = set()
    skipped_wmt22_duplicates = 0
    skipped_kde4_overlap = 0

    for pair in wmt22_pairs:
        if pair in seen_wmt22:
            skipped_wmt22_duplicates += 1
            continue

        seen_wmt22.add(pair)

        if pair in kde4_pairs:
            skipped_kde4_overlap += 1
            continue

        selected.append(pair)

        if len(selected) == requested_count:
            break

    if len(selected) < requested_count:
        raise ValueError(
            f"Fewer than {requested_count} valid WMT22 pairs are available: "
            f"selected {len(selected)}"
        )

    return selected, skipped_wmt22_duplicates, skipped_kde4_overlap


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

    kde4_all_pairs = set(kde4_train_pairs) | set(kde4_dev_pairs) | set(kde4_test_pairs)
    selected_wmt22_pairs, skipped_duplicates, skipped_kde4_overlap = select_wmt22_pairs(
        wmt22_pairs=wmt22_pairs,
        kde4_pairs=kde4_all_pairs,
        requested_count=args.num_wmt22_pairs,
    )

    final_train_pairs = kde4_train_pairs + selected_wmt22_pairs

    write_parallel_split(args.output_dir, "train", final_train_pairs)

    for split_name in ("dev", "test"):
        for suffix in ("de", "hsb"):
            copy_split_file(args.kde4_split_dir, args.output_dir, split_name, suffix)

    write_pairs(args.selected_wmt22_de, args.selected_wmt22_hsb, selected_wmt22_pairs)

    verify_split(args.output_dir, "train", len(final_train_pairs))
    verify_split(args.output_dir, "dev", len(kde4_dev_pairs))
    verify_split(args.output_dir, "test", len(kde4_test_pairs))
    verify_pair_files(
        args.selected_wmt22_de,
        args.selected_wmt22_hsb,
        len(selected_wmt22_pairs),
    )

    report_lines = [
        f"original_kde4_train_size\t{len(kde4_train_pairs)}",
        f"original_kde4_dev_size\t{len(kde4_dev_pairs)}",
        f"original_kde4_test_size\t{len(kde4_test_pairs)}",
        f"wmt22_filtered_size\t{len(wmt22_pairs)}",
        f"requested_wmt22_new_pairs\t{args.num_wmt22_pairs}",
        f"skipped_wmt22_duplicates\t{skipped_duplicates}",
        f"skipped_wmt22_pairs_overlapping_kde4_train_dev_test\t{skipped_kde4_overlap}",
        f"selected_wmt22_pairs\t{len(selected_wmt22_pairs)}",
        f"final_train_size\t{len(final_train_pairs)}",
        f"final_dev_size\t{len(kde4_dev_pairs)}",
        f"final_test_size\t{len(kde4_test_pairs)}",
        f"output_dir\t{args.output_dir}",
        f"selected_wmt22_de\t{args.selected_wmt22_de}",
        f"selected_wmt22_hsb\t{args.selected_wmt22_hsb}",
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
