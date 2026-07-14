import argparse
import csv
from itertools import zip_longest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DE_PATH = ROOT / "data" / "wmt22_parallel" / "wmt22_all.de"
DEFAULT_HSB_PATH = ROOT / "data" / "wmt22_parallel" / "wmt22_all.hsb"
DEFAULT_OUTPUT_TSV = ROOT / "outputs" / "lid_reports" / "wmt22_lid_full.tsv"
DEFAULT_BAD_TSV = ROOT / "outputs" / "lid_reports" / "wmt22_lid_bad.tsv"

GLOTLID_REPO_ID = "cis-lmu/glotlid"
GLOTLID_FILENAME = "model.bin"

EXPECTED_DE_LABEL = "__label__deu_Latn"
EXPECTED_HSB_LABEL = "__label__hsb_Latn"


HEADER = [
    "line_no",
    "status",
    "de_text",
    "hsb_text",
    "de_pred1_label",
    "de_pred1_prob",
    "de_pred2_label",
    "de_pred2_prob",
    "de_pred3_label",
    "de_pred3_prob",
    "hsb_pred1_label",
    "hsb_pred1_prob",
    "hsb_pred2_label",
    "hsb_pred2_prob",
    "hsb_pred3_label",
    "hsb_pred3_prob",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check WMT22 German-Upper Sorbian parallel data with GlotLID."
    )
    parser.add_argument("--de_path", type=Path, default=DEFAULT_DE_PATH)
    parser.add_argument("--hsb_path", type=Path, default=DEFAULT_HSB_PATH)
    parser.add_argument("--output_tsv", type=Path, default=DEFAULT_OUTPUT_TSV)
    parser.add_argument("--bad_tsv", type=Path, default=DEFAULT_BAD_TSV)
    parser.add_argument(
        "--min_prob",
        type=float,
        default=0.80,
        help="Minimum top-1 probability required for the expected language label.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of aligned sentence pairs to check.",
    )
    return parser.parse_args()


def load_glotlid_model():
    try:
        import fasttext
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install fasttext and huggingface_hub, for example:\n"
            "  pip install fasttext huggingface_hub"
        ) from exc

    model_path = hf_hub_download(repo_id=GLOTLID_REPO_ID, filename=GLOTLID_FILENAME)
    return fasttext.load_model(model_path)


def predict_top3(model, text):
    labels, probs = model.predict(text.replace("\n", " "), k=3)
    predictions = list(zip(labels, probs))

    while len(predictions) < 3:
        predictions.append(("", 0.0))

    return predictions[:3]


def side_is_bad(predictions, expected_label, min_prob):
    top_label, top_prob = predictions[0]
    return top_label != expected_label or top_prob < min_prob


def format_prob(prob):
    return f"{float(prob):.6f}"


def make_row(line_no, status, de_text, hsb_text, de_predictions, hsb_predictions):
    row = [line_no, status, de_text, hsb_text]

    for label, prob in de_predictions:
        row.extend([label, format_prob(prob)])

    for label, prob in hsb_predictions:
        row.extend([label, format_prob(prob)])

    return row


def check_files(args):
    if args.min_prob < 0.0 or args.min_prob > 1.0:
        raise ValueError("--min_prob must be between 0.0 and 1.0")

    if args.max_examples is not None and args.max_examples < 1:
        raise ValueError("--max_examples must be a positive integer when provided")

    if not args.de_path.exists():
        raise FileNotFoundError(f"German file not found: {args.de_path}")

    if not args.hsb_path.exists():
        raise FileNotFoundError(f"Upper Sorbian file not found: {args.hsb_path}")

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.bad_tsv.parent.mkdir(parents=True, exist_ok=True)

    model = load_glotlid_model()

    counts = {
        "ok": 0,
        "bad_de": 0,
        "bad_hsb": 0,
        "bad_both": 0,
    }

    with (
        args.de_path.open(encoding="utf-8", newline="") as de_file,
        args.hsb_path.open(encoding="utf-8", newline="") as hsb_file,
        args.output_tsv.open("w", encoding="utf-8", newline="") as output_file,
        args.bad_tsv.open("w", encoding="utf-8", newline="") as bad_file,
    ):
        full_writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
        bad_writer = csv.writer(bad_file, delimiter="\t", lineterminator="\n")
        full_writer.writerow(HEADER)
        bad_writer.writerow(HEADER)

        paired_lines = zip_longest(de_file, hsb_file, fillvalue=None)

        for index, (de_line, hsb_line) in enumerate(paired_lines, start=1):
            if args.max_examples is not None and index > args.max_examples:
                break

            if de_line is None or hsb_line is None:
                raise ValueError(
                    "Input files are not line-aligned: "
                    f"mismatch first seen at line {index}"
                )

            de_text = de_line.rstrip("\n\r")
            hsb_text = hsb_line.rstrip("\n\r")

            de_predictions = predict_top3(model, de_text)
            hsb_predictions = predict_top3(model, hsb_text)

            de_bad = side_is_bad(de_predictions, EXPECTED_DE_LABEL, args.min_prob)
            hsb_bad = side_is_bad(hsb_predictions, EXPECTED_HSB_LABEL, args.min_prob)

            if de_bad and hsb_bad:
                status = "bad_both"
            elif de_bad:
                status = "bad_de"
            elif hsb_bad:
                status = "bad_hsb"
            else:
                status = "ok"

            counts[status] += 1

            row = make_row(
                line_no=index,
                status=status,
                de_text=de_text,
                hsb_text=hsb_text,
                de_predictions=de_predictions,
                hsb_predictions=hsb_predictions,
            )
            full_writer.writerow(row)

            if status != "ok":
                bad_writer.writerow(row)

    return counts


def main():
    args = parse_args()
    counts = check_files(args)

    print(f"ok\t{counts['ok']}")
    print(f"bad_de\t{counts['bad_de']}")
    print(f"bad_hsb\t{counts['bad_hsb']}")
    print(f"bad_both\t{counts['bad_both']}")
    print(f"full_tsv\t{args.output_tsv}")
    print(f"bad_tsv\t{args.bad_tsv}")


if __name__ == "__main__":
    main()
