"""Create a reduced VietOnlineNews train split.

Always keeps:
- the first 500 data rows from train.csv
- any row whose id is in KEEP_IDS

Then keeps extra non-protected rows until the output has about TARGET_TOTAL_ROWS rows.
Outputs:
- train_new.csv
- train_new.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

csv.field_size_limit(2**31 - 1)

KEEP_FIRST_ROWS = 500
TARGET_TOTAL_ROWS = 10_000

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


def write_filtered_files(
    train_csv: Path,
    output_csv: Path,
    output_jsonl: Path,
    target_total_rows: int = TARGET_TOTAL_ROWS,
) -> dict[str, int]:
    rows_to_write: list[dict[str, str]] = []
    kept_first_rows = 0
    kept_id_rows = 0
    skipped_other_rows = 0
    total_rows = 0
    found_ids: set[str] = set()

    with train_csv.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {train_csv}")
        if "id" not in reader.fieldnames:
            raise ValueError(f"CSV must contain an 'id' column. Found: {reader.fieldnames}")

        fieldnames = reader.fieldnames

        for row_index, row in enumerate(reader, start=1):
            total_rows += 1
            row_id = row.get("id", "").strip()
            keep_by_position = row_index <= KEEP_FIRST_ROWS
            keep_by_id = row_id in KEEP_IDS
            protected_row = keep_by_position or keep_by_id

            if keep_by_id:
                found_ids.add(row_id)

            if protected_row:
                rows_to_write.append(row)
                if keep_by_position:
                    kept_first_rows += 1
                if keep_by_id:
                    kept_id_rows += 1
                continue

            if len(rows_to_write) < target_total_rows:
                rows_to_write.append(row)
            else:
                skipped_other_rows += 1

    with output_csv.open("w", encoding="utf-8", newline="") as csv_file, output_jsonl.open(
        "w", encoding="utf-8"
    ) as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_to_write:
            writer.writerow(row)
            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "total_rows": total_rows,
        "target_total_rows": target_total_rows,
        "kept_rows": len(rows_to_write),
        "skipped_other_rows": skipped_other_rows,
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
    parser.add_argument(
        "--target-total-rows",
        type=int,
        default=TARGET_TOTAL_ROWS,
        help="Approximate total number of rows to keep, while always preserving protected rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_csv = args.train_csv.resolve()
    output_csv = (args.output_csv or train_csv.with_name("train_new.csv")).resolve()
    output_jsonl = (args.output_jsonl or train_csv.with_name("train_new.jsonl")).resolve()

    stats = write_filtered_files(train_csv, output_csv, output_jsonl, args.target_total_rows)

    print(f"Input: {train_csv.as_posix().encode('utf-8', errors='replace').decode('ascii', errors='backslashreplace')}")
    print(f"CSV output: {output_csv.as_posix().encode('utf-8', errors='replace').decode('ascii', errors='backslashreplace')}")
    print(f"JSONL output: {output_jsonl.as_posix().encode('utf-8', errors='replace').decode('ascii', errors='backslashreplace')}")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()


