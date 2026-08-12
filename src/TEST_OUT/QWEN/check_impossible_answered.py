from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def has_generated_answer(record: dict) -> bool:
    answer = record.get("generated_answer")
    return isinstance(answer, str) and bool(answer.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find QA items marked is_possible=false in the source file "
            "that still received a non-empty generated answer."
        )
    )
    parser.add_argument(
        "--qa-file",
        type=Path,
        default=Path(__file__).with_name("data_QA_Convert.jsonl"),
        help="Path to the source QA JSONL file.",
    )
    parser.add_argument(
        "--answers-file",
        type=Path,
        default=Path(__file__).with_name("answers_qwen_colab_full.jsonl"),
        help="Path to the generated answers JSONL file.",
    )
    args = parser.parse_args()

    qa_records = load_jsonl(args.qa_file)
    answer_records = load_jsonl(args.answers_file)

    answers_by_id = {
        record.get("qa_id"): record
        for record in answer_records
        if isinstance(record.get("qa_id"), str)
    }

    matched: list[dict[str, str]] = []
    for qa in qa_records:
        if qa.get("is_possible") is not False:
            continue

        qa_id = qa.get("id")
        if not isinstance(qa_id, str):
            continue

        answer_record = answers_by_id.get(qa_id)
        if not answer_record or not has_generated_answer(answer_record):
            continue

        matched.append(
            {
                "qa_id": qa_id,
                "question": str(qa.get("question", "")),
                "generated_answer": answer_record["generated_answer"].strip(),
            }
        )

    print(f"So cau hoi is_possible=false nhung van duoc tra loi: {len(matched)}")
    print()
    for index, item in enumerate(matched, start=1):
        print(f"{index}. [{item['qa_id']}] {item['question']}")
        print(f"   Tra loi: {item['generated_answer']}")
        print()


if __name__ == "__main__":
    main()
