import argparse
import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MAX_TOKENS = 128
MAX_LENGTH_RATIO = 3.0

REASONS = [
    "empty",
    "duplicate",
    "too_long",
    "length_ratio",
    "identical",
]


def resolve_path(path: str) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return BASE_DIR / path_obj


def read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def token_count(text: str) -> int:
    return len(text.split())


def removal_reason(de: str, hsb: str, seen_pairs: set[tuple[str, str]]) -> str | None:
    if not de or not hsb:
        return "empty"

    pair = (de, hsb)
    if pair in seen_pairs:
        return "duplicate"

    de_tokens = token_count(de)
    hsb_tokens = token_count(hsb)
    if de_tokens > MAX_TOKENS or hsb_tokens > MAX_TOKENS:
        return "too_long"

    ratio = max(de_tokens / hsb_tokens, hsb_tokens / de_tokens)
    if ratio > MAX_LENGTH_RATIO:
        return "length_ratio"

    if de == hsb and len(de) > 3:
        return "identical"

    return None


def clean_pairs(src_lines: list[str], tgt_lines: list[str]):
    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"Source/reference length mismatch: {len(src_lines)} vs {len(tgt_lines)}")

    kept_de = []
    kept_hsb = []
    removed = []
    reason_counts = {reason: 0 for reason in REASONS}
    seen_pairs = set()

    for index, (raw_de, raw_hsb) in enumerate(zip(src_lines, tgt_lines)):
        de = raw_de.strip()
        hsb = raw_hsb.strip()
        reason = removal_reason(de, hsb, seen_pairs)

        if reason is not None:
            reason_counts[reason] += 1
            removed.append((index, reason, de, hsb))
            continue

        seen_pairs.add((de, hsb))
        kept_de.append(de)
        kept_hsb.append(hsb)

    return kept_de, kept_hsb, removed, reason_counts


def write_removed_pairs(path: Path, removed: list[tuple[int, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["index", "reason", "de", "hsb"])
        writer.writerows(removed)


def format_report(total: int, kept: int, removed: int, reason_counts: dict[str, int]) -> str:
    kept_percent = 100.0 * kept / total if total else 0.0
    lines = [
        f"total input pairs: {total}",
        f"kept pairs: {kept}",
        f"removed pairs: {removed}",
    ]
    for reason in REASONS:
        lines.append(f"removed {reason}: {reason_counts[reason]}")
    lines.append(f"percentage kept: {kept_percent:.2f}%")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Clean KDE4 de-hsb parallel data.")
    parser.add_argument("--src_path", default="../data/KDE4.de-hsb.de")
    parser.add_argument("--ref_path", default="../data/KDE4.de-hsb.hsb")
    parser.add_argument("--output_dir", default="../data/cleaned")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src_lines = read_lines(resolve_path(args.src_path))
    tgt_lines = read_lines(resolve_path(args.ref_path))

    kept_de, kept_hsb, removed, reason_counts = clean_pairs(src_lines, tgt_lines)

    write_lines(output_dir / "KDE4.cleaned.de", kept_de)
    write_lines(output_dir / "KDE4.cleaned.hsb", kept_hsb)
    write_removed_pairs(output_dir / "removed_pairs.tsv", removed)

    report = format_report(
        total=len(src_lines),
        kept=len(kept_de),
        removed=len(removed),
        reason_counts=reason_counts,
    )
    with open(output_dir / "cleaning_report.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(report)
    print("Saved cleaned data to:", output_dir)


if __name__ == "__main__":
    main()
