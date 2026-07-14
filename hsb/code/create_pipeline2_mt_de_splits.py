import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KDE4_SPLIT_DIR = ROOT / "data" / "splits"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create KDE4 train plus filtered Pipeline 2 synthetic MT-German/hsb splits."
    )
    parser.add_argument("--kde4_split_dir", type=Path, default=DEFAULT_KDE4_SPLIT_DIR)
    parser.add_argument("--pipeline2_de_path", required=True, type=Path)
    parser.add_argument("--pipeline2_hsb_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--report_path", required=True, type=Path)
    return parser.parse_args()


def read_lines(path):
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8-sig") as handle:
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
    output_dir.mkdir(parents=True, exist_ok=True)
    de_path = output_dir / f"{split_name}.de"
    hsb_path = output_dir / f"{split_name}.hsb"

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


def filter_pipeline2_pairs(pipeline2_pairs, kde4_pairs):
    seen_pipeline2 = set()
    kept_pairs = []
    duplicates_removed = 0
    overlap_removed = 0

    for pair in pipeline2_pairs:
        if pair in seen_pipeline2:
            duplicates_removed += 1
            continue
        seen_pipeline2.add(pair)

        if pair in kde4_pairs:
            overlap_removed += 1
            continue

        kept_pairs.append(pair)

    return kept_pairs, duplicates_removed, overlap_removed


def write_report(report_path, report_items):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}\t{value}" for key, value in report_items]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    print(f"report_path\t{report_path}")


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
    pipeline2_pairs = read_parallel(args.pipeline2_de_path, args.pipeline2_hsb_path)

    kde4_all_pairs = set(kde4_train_pairs) | set(kde4_dev_pairs) | set(kde4_test_pairs)
    filtered_pipeline2_pairs, duplicates_removed, overlap_removed = filter_pipeline2_pairs(
        pipeline2_pairs,
        kde4_all_pairs,
    )

    final_train_pairs = kde4_train_pairs + filtered_pipeline2_pairs

    write_parallel(args.output_dir, "train", final_train_pairs)
    for split_name in ("dev", "test"):
        for suffix in ("de", "hsb"):
            copy_split_file(args.kde4_split_dir, args.output_dir, split_name, suffix)

    verify_split(args.output_dir, "train", len(final_train_pairs))
    verify_split(args.output_dir, "dev", len(kde4_dev_pairs))
    verify_split(args.output_dir, "test", len(kde4_test_pairs))

    report_items = [
        ("original_kde4_train_size", len(kde4_train_pairs)),
        ("original_kde4_dev_size", len(kde4_dev_pairs)),
        ("original_kde4_test_size", len(kde4_test_pairs)),
        ("pipeline2_input_size", len(pipeline2_pairs)),
        ("pipeline2_duplicates_removed", duplicates_removed),
        ("pipeline2_overlap_with_kde4_removed", overlap_removed),
        ("pipeline2_kept_size", len(filtered_pipeline2_pairs)),
        ("final_train_size", len(final_train_pairs)),
        ("final_dev_size", len(kde4_dev_pairs)),
        ("final_test_size", len(kde4_test_pairs)),
    ]
    write_report(args.report_path, report_items)


def main():
    args = parse_args()
    create_splits(args)


if __name__ == "__main__":
    main()
