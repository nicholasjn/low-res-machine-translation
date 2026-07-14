import argparse
import csv
import gzip
import re
from collections import defaultdict
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
    "sk": "sk",
    "slk": "sk",
    "de": "de",
    "deu": "de",
    "ger": "de",
}

SUPPORTED_SUFFIXES = {
    ".gz",
    ".txt",
    ".tsv",
    ".en",
    ".eng",
    ".cs",
    ".ces",
    ".cz",
    ".pl",
    ".pol",
    ".dsb",
    ".hsb",
}

NON_DE_LANGS = {"dsb", "en", "cs", "pl", "sk"}
LANG_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(" + "|".join(sorted(LANG_ALIASES, key=len, reverse=True)) + r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count likely non-German Upper Sorbian parallel data candidates."
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
        current = Path(name)
        suffix = current.suffix.lower()
        if suffix in SUPPORTED_SUFFIXES:
            name = current.stem
        else:
            break

    parts = re.split(r"([._\-\s]+)", name)
    stripped_parts = []
    for part in parts:
        if part.lower() in LANG_ALIASES:
            continue
        stripped_parts.append(part)

    key = "".join(stripped_parts).strip("._- ")
    key = re.sub(r"[._\-\s]+", "_", key)
    return key or path.stem.lower()


def line_count(path):
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        return sum(1 for _ in handle)


def choose_file(files):
    return sorted(files, key=lambda item: (len(item["languages"]), str(item["path"])))[0]


def scan_groups(input_dir):
    groups = defaultdict(list)

    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or not has_supported_suffix(path):
            continue

        languages = detect_languages(path.name)
        if "hsb" not in languages and not (set(languages) & NON_DE_LANGS):
            continue

        groups[group_key(path)].append(
            {
                "path": path,
                "languages": languages,
            }
        )

    return groups


def build_rows(groups):
    rows = []
    line_cache = {}

    for key, files in sorted(groups.items()):
        hsb_files = [item for item in files if "hsb" in item["languages"]]
        if not hsb_files:
            continue

        for src_lang in sorted(NON_DE_LANGS):
            src_files = [
                item
                for item in files
                if src_lang in item["languages"] and "de" not in item["languages"]
            ]
            if not src_files:
                continue

            hsb_file = choose_file(hsb_files)
            src_file = choose_file(src_files)
            if hsb_file["path"] == src_file["path"]:
                continue

            for item in (hsb_file, src_file):
                path = item["path"]
                if path not in line_cache:
                    line_cache[path] = line_count(path)

            hsb_lines = line_cache[hsb_file["path"]]
            src_lines = line_cache[src_file["path"]]
            min_pairs = min(hsb_lines, src_lines)
            notes = []
            if hsb_lines != src_lines:
                notes.append("line_count_mismatch")
            if len(hsb_files) > 1:
                notes.append(f"multiple_hsb_files={len(hsb_files)}")
            if len(src_files) > 1:
                notes.append(f"multiple_{src_lang}_files={len(src_files)}")
            if any("de" in item["languages"] for item in files):
                notes.append("group_also_contains_de_file")
            if src_lang == "dsb":
                notes.append("related_sorbian_source")

            rows.append(
                {
                    "group_key": key,
                    "lang_pair": f"{src_lang}-hsb",
                    "src_lang": src_lang,
                    "hsb_file": str(hsb_file["path"]),
                    "src_file": str(src_file["path"]),
                    "hsb_lines": hsb_lines,
                    "src_lines": src_lines,
                    "min_pairs": min_pairs,
                    "enough_for_100k": "yes" if min_pairs >= 100000 else "no",
                    "notes": ";".join(notes),
                }
            )

    return sorted(rows, key=lambda row: (-row["min_pairs"], row["group_key"], row["src_lang"]))


def write_tsv(rows, output_tsv):
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "group_key",
        "lang_pair",
        "src_lang",
        "hsb_file",
        "src_file",
        "hsb_lines",
        "src_lines",
        "min_pairs",
        "enough_for_100k",
        "notes",
    ]
    with output_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print("group_key\tlang_pair\tmin_pairs\tenough_for_100k\tnotes")
    for row in rows:
        print(
            f"{row['group_key']}\t{row['lang_pair']}\t{row['min_pairs']}\t"
            f"{row['enough_for_100k']}\t{row['notes']}"
        )
    print(f"total_candidate_groups\t{len(rows)}")
    print(f"groups_enough_for_100k\t{sum(1 for row in rows if row['enough_for_100k'] == 'yes')}")


def main():
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    groups = scan_groups(args.input_dir)
    rows = build_rows(groups)
    write_tsv(rows, args.output_tsv)
    print_summary(rows)
    print(f"output_tsv\t{args.output_tsv}")


if __name__ == "__main__":
    main()
