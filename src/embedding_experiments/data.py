from __future__ import annotations

import ast
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

try:
    from src.data_ingestion.preprocess import PreprocessConfig, preprocess_article, record_to_json
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.data_ingestion.preprocess import PreprocessConfig, preprocess_article, record_to_json


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    question: str
    relevant_article_ids: set[str]
    qa_type: str


def read_jsonl(path: str | Path) -> Iterator[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, object]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def preprocess_corpus_csvs(
    input_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    min_content_chars: int = 300,
    long_content_chars: int = 20_000,
) -> dict[str, int]:
    config = PreprocessConfig(
        min_content_chars=min_content_chars,
        long_content_chars=long_content_chars,
        strip_urls=False,
        include_embedding_text=True,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "files_read": 0,
        "rows_read": 0,
        "rows_written": 0,
        "duplicate_article_ids": 0,
        "missing_required": 0,
    }
    seen_ids: set[str] = set()
    with output_path.open("w", encoding="utf-8", newline="") as target:
        for input_path in input_paths:
            input_path = Path(input_path)
            if not input_path.exists():
                continue
            stats["files_read"] += 1
            split = input_path.stem.replace("_new", "")
            with input_path.open("r", encoding="utf-8-sig", newline="") as source:
                reader = csv.DictReader(source)
                for row in reader:
                    stats["rows_read"] += 1
                    if any(not row.get(column) for column in ("id", "title", "description", "content", "category")):
                        stats["missing_required"] += 1
                        continue
                    article_id = str(row.get("id", "")).strip()
                    if article_id in seen_ids:
                        stats["duplicate_article_ids"] += 1
                        continue
                    seen_ids.add(article_id)
                    record = preprocess_article(row, config)
                    payload = record_to_json(record)
                    payload["metadata"] = {**dict(payload.get("metadata") or {}), "split": split}
                    target.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                    target.write("\n")
                    stats["rows_written"] += 1
    return stats


def preprocess_corpus_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    min_content_chars: int = 300,
    long_content_chars: int = 20_000,
    split: str = "train",
) -> dict[str, int]:
    config = PreprocessConfig(
        min_content_chars=min_content_chars,
        long_content_chars=long_content_chars,
        strip_urls=False,
        include_embedding_text=True,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "files_read": 1,
        "rows_read": 0,
        "rows_written": 0,
        "duplicate_article_ids": 0,
        "missing_required": 0,
    }
    seen_ids: set[str] = set()
    with Path(input_path).open("r", encoding="utf-8-sig") as source, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        for line in source:
            line = line.strip()
            if not line:
                continue
            stats["rows_read"] += 1
            row = json.loads(line)
            normalized_row = {
                "id": row.get("id") or row.get("article_id"),
                "title": row.get("title"),
                "description": row.get("description"),
                "content": row.get("content"),
                "category": row.get("category"),
            }
            if any(not normalized_row.get(column) for column in ("id", "title", "description", "content", "category")):
                stats["missing_required"] += 1
                continue
            article_id = str(normalized_row["id"]).strip()
            if article_id in seen_ids:
                stats["duplicate_article_ids"] += 1
                continue
            seen_ids.add(article_id)
            record = preprocess_article(normalized_row, config)
            payload = record_to_json(record)
            payload["metadata"] = {**dict(payload.get("metadata") or {}), "split": split}
            target.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            target.write("\n")
            stats["rows_written"] += 1
    return stats


def load_chunks(path: str | Path) -> list[dict[str, object]]:
    return list(read_jsonl(path))


def load_queries(path: str | Path, *, include_impossible: bool = False, limit: int | None = None) -> list[QueryRecord]:
    queries: list[QueryRecord] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if limit is not None and len(queries) >= limit:
                break
            is_possible = _as_bool(row.get("is_possible"))
            if not is_possible and not include_impossible:
                continue
            relevant_ids = _relevant_ids(row)
            question = str(row.get("question") or "").strip()
            if not question or not relevant_ids:
                continue
            queries.append(
                QueryRecord(
                    query_id=str(row.get("id") or len(queries)),
                    question=question,
                    relevant_article_ids=relevant_ids,
                    qa_type=str(row.get("qa_type") or "unknown"),
                )
            )
    return queries


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _relevant_ids(row: dict[str, str]) -> set[str]:
    relevant = {str(row.get("article_id") or "").strip()}
    source_ids = str(row.get("source_article_ids") or "").strip()
    if source_ids:
        parsed = _parse_source_ids(source_ids)
        relevant.update(parsed)
    return {item for item in relevant if item and item.lower() != "nan"}


def _parse_source_ids(value: str) -> set[str]:
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple, set)):
            return {str(item).strip() for item in parsed if str(item).strip()}
    except (SyntaxError, ValueError):
        pass
    return {item.strip().strip("'\"") for item in value.replace(";", ",").split(",") if item.strip()}

