from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def result_from_chunk(row: dict[str, object], score: float) -> dict[str, object]:
    metadata = dict(row.get("metadata") or {})
    return {
        "chunk_id": str(row["chunk_id"]),
        "article_id": str(row["article_id"]),
        "score": float(score),
        "text": str(row.get("text") or ""),
        "chunk_text": str(row.get("chunk_text") or ""),
        "title": metadata.get("title"),
        "category": metadata.get("category"),
        "chunk_index": metadata.get("chunk_index"),
    }
