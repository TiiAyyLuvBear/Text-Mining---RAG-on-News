"""Concatenate the two QA JSONL files in a fixed order."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
QA_DIR = BASE_DIR.parent / "QA_Claude"

INPUT_PATHS = [
    QA_DIR / "QA_output.jsonl",
    QA_DIR / "QA_output_new_480.jsonl",
]
OUTPUT_PATH = BASE_DIR / "data_QA_Convert.jsonl"


def merge_jsonl(input_paths: list[Path], output_path: Path) -> list[int]:
    """Validate and concatenate JSONL inputs without changing record content."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    counts = []

    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
            for input_path in input_paths:
                count = 0
                with input_path.open("r", encoding="utf-8") as source:
                    for line_number, line in enumerate(source, start=1):
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            json.loads(stripped)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"Invalid JSON at {input_path}:{line_number}: {exc}"
                            ) from exc
                        output.write(stripped + "\n")
                        count += 1
                counts.append(count)

            output.flush()
            os.fsync(output.fileno())

        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return counts


def main() -> None:
    for input_path in INPUT_PATHS:
        if not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")

    counts = merge_jsonl(INPUT_PATHS, OUTPUT_PATH)
    for input_path, count in zip(INPUT_PATHS, counts):
        print(f"{input_path.name}: {count} records")
    print(f"Total: {sum(counts)} records")
    print(f"Saved: {ascii(str(OUTPUT_PATH))}")


if __name__ == "__main__":
    main()
