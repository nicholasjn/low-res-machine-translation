import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
XCOMET_DIR = ROOT / "outputs" / "xcomet"
SUMMARY_PATH = XCOMET_DIR / "xcomet_summary.tsv"


def main():
    rows = []
    for json_path in sorted(XCOMET_DIR.glob("*_xcomet.json")):
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        system_score = data.get("system_score", "")
        n = data.get("n")
        if n is None:
            n = len(data.get("sentence_scores", []))
        rows.append((str(json_path), system_score, n))

    XCOMET_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", encoding="utf-8", newline="") as handle:
        handle.write("system_file\tsystem_score\tn\n")
        for system_file, system_score, n in rows:
            handle.write(f"{system_file}\t{system_score}\t{n}\n")

    print("system_file\tsystem_score\tn")
    for system_file, system_score, n in rows:
        print(f"{system_file}\t{system_score}\t{n}")
    print(f"\nSaved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
