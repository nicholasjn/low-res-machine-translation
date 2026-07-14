import argparse
import json
from pathlib import Path


def read_lines(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def get_result_value(result, key, default=None):
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MT output with xCOMET.")
    parser.add_argument("--src", required=True, help="Source text file.")
    parser.add_argument("--mt", required=True, help="MT hypothesis file.")
    parser.add_argument("--ref", required=True, help="Reference text file.")
    parser.add_argument("--model_name", default="Unbabel/XCOMET-XL")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_tsv", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as exc:
        raise SystemExit(
            "Could not import unbabel-comet. Install it with:\n"
            'python3 -m pip install --user "unbabel-comet>=2.2.0"'
        ) from exc

    src_lines = read_lines(args.src)
    mt_lines = read_lines(args.mt)
    ref_lines = read_lines(args.ref)

    n = min(len(src_lines), len(mt_lines), len(ref_lines))
    if args.num_samples > 0:
        n = min(n, args.num_samples)

    src_lines = src_lines[:n]
    mt_lines = mt_lines[:n]
    ref_lines = ref_lines[:n]

    data = [
        {"src": src, "mt": mt, "ref": ref}
        for src, mt, ref in zip(src_lines, mt_lines, ref_lines)
    ]

    print(f"Source: {args.src}")
    print(f"MT: {args.mt}")
    print(f"Reference: {args.ref}")
    print(f"Examples: {n}")
    print(f"Model: {args.model_name}")

    model_path = download_model(args.model_name)
    model = load_from_checkpoint(model_path)
    result = model.predict(data, batch_size=args.batch_size, gpus=args.gpus)

    scores = get_result_value(result, "scores", [])
    system_score = get_result_value(result, "system_score")
    if system_score is None and scores:
        system_score = sum(scores) / len(scores)

    sentence_rows = []
    for idx, (src, mt, ref, score) in enumerate(zip(src_lines, mt_lines, ref_lines, scores), 1):
        sentence_rows.append(
            {
                "index": idx,
                "score": float(score),
                "src": src,
                "mt": mt,
                "ref": ref,
            }
        )

    output_json = Path(args.output_json)
    output_tsv = Path(args.output_tsv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_name": args.model_name,
        "src": args.src,
        "mt": args.mt,
        "ref": args.ref,
        "n": n,
        "system_score": None if system_score is None else float(system_score),
        "sentence_scores": sentence_rows,
    }

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with output_tsv.open("w", encoding="utf-8", newline="") as handle:
        handle.write("index\tscore\tsrc\tmt\tref\n")
        for row in sentence_rows:
            handle.write(
                f"{row['index']}\t{row['score']}\t"
                f"{row['src'].replace(chr(9), ' ')}\t"
                f"{row['mt'].replace(chr(9), ' ')}\t"
                f"{row['ref'].replace(chr(9), ' ')}\n"
            )

    print(f"JSON: {output_json}")
    print(f"TSV: {output_tsv}")
    print(f"xCOMET system score: {system_score}")


if __name__ == "__main__":
    main()
