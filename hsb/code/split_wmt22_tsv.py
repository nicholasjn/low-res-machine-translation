from pathlib import Path
import csv

def split_tsv(tsv_path, de_out, hsb_out):
    de_lines = []
    hsb_lines = []

    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue

            # HSB-DE_train/dev.tsv: 第一列是 hsb，第二列是 de
            hsb = row[0].strip()
            de = row[1].strip()

            if de and hsb:
                hsb_lines.append(hsb)
                de_lines.append(de)

    Path(de_out).write_text("\n".join(de_lines) + "\n", encoding="utf-8")
    Path(hsb_out).write_text("\n".join(hsb_lines) + "\n", encoding="utf-8")
    print(tsv_path, "->", len(de_lines), "pairs")

split_tsv("HSB-DE_train.tsv", "train_2022.de", "train_2022.hsb")
split_tsv("HSB-DE_dev.tsv", "dev_2022.de", "dev_2022.hsb")