import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


LANG_ALIASES = {
    "hsb": "hsb",
    "dsb": "dsb",
    "en": "en",
    "eng": "en",
    "cs": "cs",
    "ces": "cs",
    "cz": "cs",
    "pl": "pl",
    "pol": "pl",
    "de": "de",
    "deu": "de",
    "ger": "de",
}

SUPPORTED_SUFFIXES = {
    ".gz",
    ".xz",
    ".zip",
    ".tsv",
    ".txt",
    ".de",
    ".hsb",
    ".en",
    ".cs",
    ".pl",
    ".dsb",
}

LANG_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(" + "|".join(sorted(LANG_ALIASES, key=len, reverse=True)) + r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inventory possible Upper Sorbian multilingual or parallel source files."
    )
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_tsv", required=True, type=Path)
    return parser.parse_args()


def has_supported_suffix(path):
    return any(suffix.lower() in SUPPORTED_SUFFIXES for suffix in path.suffixes)


def detect_languages(filename):
    languages = {LANG_ALIASES[match.group(1).lower()] for match in LANG_PATTERN.finditer(filename)}

    for suffix in Path(filename).suffixes:
        suffix_lang = suffix.lower().lstrip(".")
        if suffix_lang in LANG_ALIASES:
            languages.add(LANG_ALIASES[suffix_lang])

    return sorted(languages)


def group_key(path):
    name = path.name.lower()

    while True:
        stem, suffix = Path(name).stem, Path(name).suffix.lower()
        if suffix in SUPPORTED_SUFFIXES:
            name = stem
        else:
            break

    parts = re.split(r"([._\-\s]+)", name)
    stripped_parts = []
    for part in parts:
        normalized = part.lower()
        if normalized in LANG_ALIASES:
            continue
        stripped_parts.append(part)

    key = "".join(stripped_parts).strip("._- ")
    key = re.sub(r"[._\-\s]+", "_", key)
    return key or path.stem.lower()


def make_notes(path, languages, groups):
    notes = []
    if "hsb" in languages:
        notes.append("contains_hsb")
    if "de" in languages:
        notes.append("contains_de")
    if len(languages) > 1:
        notes.append("multilingual_filename")
    if len(groups[group_key(path)]) > 1:
        notes.append("possible_parallel_group")
    if not has_supported_suffix(path):
        notes.append("unsupported_suffix_candidate_by_name")
    return ";".join(notes)


def scan_files(input_dir):
    groups = defaultdict(list)
    rows = []

    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue

        languages = detect_languages(path.name)
        if not languages and not has_supported_suffix(path):
            continue
        if not languages:
            continue

        key = group_key(path)
        row = {
            "path": path,
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "detected_languages": ",".join(languages),
            "possible_group_key": key,
            "languages": languages,
        }
        rows.append(row)
        groups[key].append(row)

    return rows, groups


def write_tsv(rows, groups, output_tsv):
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "file_path",
                "filename",
                "size_bytes",
                "detected_languages",
                "possible_group_key",
                "notes",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    str(row["path"]),
                    row["filename"],
                    row["size_bytes"],
                    row["detected_languages"],
                    row["possible_group_key"],
                    make_notes(row["path"], row["languages"], groups),
                ]
            )


def summarize(rows, groups):
    combo_counts = Counter(row["detected_languages"] for row in rows)
    hsb_not_de = sum(1 for row in rows if "hsb" in row["languages"] and "de" not in row["languages"])
    hsb_and_de = sum(1 for row in rows if {"hsb", "de"}.issubset(row["languages"]))

    print(f"total candidate files\t{len(rows)}")
    print("candidate language combinations")
    for combo, count in sorted(combo_counts.items()):
        print(f"{combo}\t{count}")
    print(f"files with hsb but not de\t{hsb_not_de}")
    print(f"files with hsb and de\t{hsb_and_de}")

    parallel_groups = [
        (key, sorted({lang for row in group_rows for lang in row["languages"]}))
        for key, group_rows in groups.items()
        if len(group_rows) > 1
    ]
    if parallel_groups:
        print("possible parallel groups")
        for key, languages in sorted(parallel_groups):
            print(f"{key}\t{','.join(languages)}")


def main():
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    rows, groups = scan_files(args.input_dir)
    write_tsv(rows, groups, args.output_tsv)
    summarize(rows, groups)
    print(f"output_tsv\t{args.output_tsv}")


if __name__ == "__main__":
    main()
