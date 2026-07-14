import argparse
import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = "../results/metrics.csv"

FIELDS = [
    "direction",
    "model",
    "method",
    "stage",
    "num_test",
    "BLEU",
    "chrF++",
    "output_file",
    "notes",
]

DIRECTION_LABELS = {
    "de-hsb": "de -> hsb",
    "hsb-de": "hsb -> de",
}

ALLOWED_DIRECTIONS = ["de-hsb", "hsb-de"]
ALLOWED_STAGES = ["debug", "full"]
ALLOWED_METHODS = ["baseline", "extended-baseline", "finetune", "lora"]


def resolve_path(path: str) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return BASE_DIR / path_obj


def ensure_csv(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()


def format_score(value: str) -> str:
    try:
        return f"{float(value):.2f}"
    except ValueError:
        return value


def add_result(args) -> None:
    csv_path = resolve_path(args.csv_path)
    ensure_csv(csv_path)

    row = {
        "direction": args.direction,
        "model": args.model,
        "method": args.method,
        "stage": args.stage,
        "num_test": str(args.num_test),
        "BLEU": format_score(str(args.bleu)),
        "chrF++": format_score(str(args.chrf)),
        "output_file": args.output_file,
        "notes": args.notes,
    }

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writerow(row)

    print("Added result to:", csv_path)
    print(row)


def read_results(csv_path: Path) -> list[dict]:
    ensure_csv(csv_path)
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def markdown_escape(value: str) -> str:
    return (value or "").replace("|", "\\|")


def print_direction_table(direction: str, rows: list[dict]) -> None:
    print(f"## Direction: {DIRECTION_LABELS[direction]}")
    print()
    print("| model | method | stage | num_test | BLEU | chrF++ | output_file | notes |")
    print("|---|---|---:|---:|---:|---:|---|---|")

    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    markdown_escape(row["model"]),
                    markdown_escape(row["method"]),
                    markdown_escape(row["stage"]),
                    markdown_escape(row["num_test"]),
                    markdown_escape(row["BLEU"]),
                    markdown_escape(row["chrF++"]),
                    markdown_escape(row["output_file"]),
                    markdown_escape(row["notes"]),
                ]
            )
            + " |"
        )
    print()


def show_results(args) -> None:
    csv_path = resolve_path(args.csv_path)
    rows = read_results(csv_path)

    for direction in ALLOWED_DIRECTIONS:
        direction_rows = [row for row in rows if row.get("direction") == direction]
        print_direction_table(direction, direction_rows)

    print("CSV:", csv_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Record and display MT experiment metrics. Official Qwen results should use "
            "Qwen/Qwen2.5-7B-Instruct; smaller Qwen checkpoints are intended only for local smoke tests."
        )
    )
    parser.add_argument("--csv_path", default=DEFAULT_CSV_PATH)

    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add one experiment result.")
    add_parser.add_argument("--direction", choices=ALLOWED_DIRECTIONS, required=True)
    add_parser.add_argument("--model", required=True)
    add_parser.add_argument("--method", choices=ALLOWED_METHODS, required=True)
    add_parser.add_argument("--stage", choices=ALLOWED_STAGES, required=True)
    add_parser.add_argument("--num_test", type=int, required=True)
    add_parser.add_argument("--bleu", type=float, required=True)
    add_parser.add_argument("--chrf", type=float, required=True)
    add_parser.add_argument("--output_file", required=True)
    add_parser.add_argument("--notes", default="")
    add_parser.set_defaults(func=add_result)

    show_parser = subparsers.add_parser("show", help="Show grouped markdown tables.")
    show_parser.set_defaults(func=show_results)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
