"""Convert a retrieval evaluation CSV into BGE reranker JSONL inputs.

Each output preserves one retrieval strategy's ordered candidate pool and adds
the chunk text required by a cross-encoder reranker.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


STRATEGIES = ("bm25", "dense", "hybrid")


def load_chunks(chunks_path: Path) -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    with chunks_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                chunks[str(row["chunk_id"])] = row
    return chunks


def build_inputs(report_path: Path, chunks_path: Path) -> dict[str, list[dict[str, Any]]]:
    chunks = load_chunks(chunks_path)
    outputs: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with report_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            strategy = row["method"]
            if strategy not in STRATEGIES:
                continue
            candidate_ids = json.loads(row["retrieved_chunk_ids"])
            candidates = []
            for rank, chunk_id in enumerate(candidate_ids, start=1):
                chunk = chunks.get(str(chunk_id))
                if chunk is None:
                    raise KeyError(f"Chunk {chunk_id} from {row['qa_id']} is absent from {chunks_path}.")
                candidates.append(
                    {
                        "rank": rank,
                        "chunk_id": str(chunk_id),
                        "article_id": str(chunk["article_id"]),
                        "text": str(chunk.get("text") or chunk.get("chunk_text") or ""),
                    }
                )
            if "relevant_chunk_ids" not in row:
                raise ValueError(
                    "Reranking requires chunk-level retrieval results with "
                    "a relevant_chunk_ids column. Re-run retrieval using gold_id qrels."
                )
            outputs[strategy].append(
                {
                    "qa_id": row["qa_id"],
                    "strategy": strategy,
                    "question": row["question"],
                    "is_possible": row.get("is_possible", "True") == "True",
                    "gold_chunk_ids": json.loads(row["relevant_chunk_ids"]),
                    "candidate_k": len(candidates),
                    "candidates": candidates,
                }
            )

    for strategy in STRATEGIES:
        rows = outputs[strategy]
        if len({row["qa_id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate QA IDs in {strategy} rows.")
    return outputs


def write_inputs(inputs: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for strategy, rows in inputs.items():
        output_path = output_dir / f"{strategy}_top50.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} queries to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create one BGE input JSONL per retrieval strategy.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_inputs(build_inputs(args.report, args.chunks), args.output_dir)


if __name__ == "__main__":
    main()
