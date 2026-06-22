"""Create a reduced VietOnlineNews train split.

Keeps:
- the first 500 data rows from train.csv
- any row whose id is in KEEP_IDS

Outputs:
- train_new.csv
- train_new.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

KEEP_FIRST_ROWS = 500

KEEP_IDS = {
    "150854", "150858", "150614",
    "152478", "152549", "153051",
    "79595", "75539", "78693",
    "79333", "79060", "78375",
    "178976", "178937", "177700",
    "102777", "178328", "4420",
    "90555", "164317", "91968",
    "15768", "15473", "168961",
    "54620", "187611", "54066",
    "183632", "184630", "183453",
    "111170", "111966", "111997",
    "109851", "113767", "111756",
    "26619", "26887", "204720",
    "204015", "204035", "26707",
    "116986", "116279", "116533",
    "208005", "205316", "31675",
    "122604", "120599", "121437",
    "217822", "218093", "121509",
    "221769", "222088", "219358",
    "221710", "37169", "225971",
    "140608", "133519", "244381",
    "139413", "247390", "133992",
    "145672", "52000", "144527",
    "254260", "143170", "146482",
    "70126", "71542", "70125",
    "264833", "265189", "264994",
}


def default_train_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "Dataset"
        / "Create_QA_Vietonline"
        / "VietOnlineNews_CSV"
        / "train.csv"
    )


def write_filtered_files(train_csv: Path, output_csv: Path, output_jsonl: Path) -> dict[str, int]:
    kept_rows = 0
    kept_first_rows = 0
    kept_id_rows = 0
    total_rows = 0
    found_ids: set[str] = set()

    with train_csv.open("r", encoding="utf-8-sig", newline="") as input_file, output_csv.open(
        "w", encoding="utf-8", newline=""
    ) as csv_file, output_jsonl.open("w", encoding="utf-8") as jsonl_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {train_csv}")
        if "id" not in reader.fieldnames:
            raise ValueError(f"CSV must contain an 'id' column. Found: {reader.fieldnames}")

        writer = csv.DictWriter(csv_file, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row_index, row in enumerate(reader, start=1):
            total_rows += 1
            row_id = row.get("id", "").strip()
            keep_by_position = row_index <= KEEP_FIRST_ROWS
            keep_by_id = row_id in KEEP_IDS

            if not keep_by_position and not keep_by_id:
                continue

            writer.writerow(row)
            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept_rows += 1

            if keep_by_position:
                kept_first_rows += 1
            if keep_by_id:
                kept_id_rows += 1
                found_ids.add(row_id)

    return {
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "kept_first_rows": kept_first_rows,
        "kept_id_rows": kept_id_rows,
        "unique_requested_ids": len(KEEP_IDS),
        "unique_found_ids": len(found_ids),
        "missing_requested_ids": len(KEEP_IDS - found_ids),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter train.csv into train_new.csv and train_new.jsonl")
    parser.add_argument("--train-csv", type=Path, default=default_train_path())
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_csv = args.train_csv.resolve()
    output_csv = (args.output_csv or train_csv.with_name("train_new.csv")).resolve()
    output_jsonl = (args.output_jsonl or train_csv.with_name("train_new.jsonl")).resolve()

    stats = write_filtered_files(train_csv, output_csv, output_jsonl)

    print(f"Input: {train_csv}")
    print(f"CSV output: {output_csv}")
    print(f"JSONL output: {output_jsonl}")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

