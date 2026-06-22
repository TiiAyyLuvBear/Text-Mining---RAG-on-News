"""Create reduced VietOnlineNews test and validation splits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

csv.field_size_limit(2**31 - 1)

DEFAULT_LIMITS = dict(test=2000, validation=2500)


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "Dataset" / "Create_QA_Vietonline" / "VietOnlineNews"


def write_limited_split(input_csv: Path, output_csv: Path, output_jsonl: Path, limit: int) -> dict[str, int]:
    total_rows = 0
    kept_rows = 0
    kept_ids: set[str] = set()

    with input_csv.open("r", encoding="utf-8-sig", newline="") as input_file, output_csv.open("w", encoding="utf-8", newline="") as csv_file, output_jsonl.open("w", encoding="utf-8") as jsonl_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header: " + str(input_csv))
        if "id" not in reader.fieldnames:
            raise ValueError("CSV must contain an id column. Found: " + str(reader.fieldnames))

        writer = csv.DictWriter(csv_file, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            total_rows += 1
            if kept_rows >= limit:
                continue

            writer.writerow(row)
            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept_rows += 1
            kept_ids.add(row.get("id", "").strip())

    return dict(total_rows=total_rows, target_rows=limit, kept_rows=kept_rows, unique_kept_ids=len(kept_ids))


def safe_path(path: Path) -> str:
    return path.as_posix().encode("utf-8", errors="replace").decode("ascii", errors="backslashreplace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create reduced test and validation CSV/JSONL files")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--test-limit", type=int, default=DEFAULT_LIMITS["test"])
    parser.add_argument("--validation-limit", type=int, default=DEFAULT_LIMITS["validation"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    split_limits = dict(test=args.test_limit, validation=args.validation_limit)

    for split_name, limit in split_limits.items():
        input_csv = data_dir / (split_name + ".csv")
        output_csv = data_dir / (split_name + "_new.csv")
        output_jsonl = data_dir / (split_name + "_new.jsonl")
        stats = write_limited_split(input_csv, output_csv, output_jsonl, limit)

        print(split_name + " input: " + safe_path(input_csv))
        print(split_name + " CSV output: " + safe_path(output_csv))
        print(split_name + " JSONL output: " + safe_path(output_jsonl))
        for key, value in stats.items():
            print(split_name + " " + key + ": " + str(value))


if __name__ == "__main__":
    main()

