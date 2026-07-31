from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Collection, Protocol

from .schema import read_jsonl, write_jsonl
from .tokenize import tokenize_vietnamese


class SemanticEncoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: object) -> object: ...


def lexical_overlap(answer: str, chunk_text: str) -> float:
    answer_terms = set(tokenize_vietnamese(answer))
    chunk_terms = set(tokenize_vietnamese(chunk_text))
    return len(answer_terms & chunk_terms) / len(answer_terms) if answer_terms else 0.0


def article_ids_from_qa(qa: dict[str, object]) -> list[str]:
    """Return source article IDs from scalar, list, or JSON-list QA fields."""
    values: list[object] = []
    for field in ("source_article_ids", "article_id"):
        raw = qa.get(field)
        if raw is None:
            continue
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = raw
            raw = parsed
        values.extend(raw if isinstance(raw, (list, tuple, set)) else [raw])

    ids: list[str] = []
    for value in values:
        article_id = str(value).strip()
        if article_id and article_id not in ids:
            ids.append(article_id)
    return ids


def build_weak_qrels(
    qa_path: str | Path,
    chunks_path: str | Path,
    *,
    semantic_encoder: SemanticEncoder | None = None,
    relevance_tolerance: float = 0.02,
    show_progress: bool = False,
    skip_qa_ids: Collection[str] | None = None,
) -> list[dict[str, object]]:
    """Map answers to their best matching source chunks; labels are explicitly weak."""
    chunks_by_article: dict[str, list[dict[str, object]]] = defaultdict(list)
    for chunk in read_jsonl(chunks_path):
        chunks_by_article[str(chunk["article_id"])].append(chunk)

    qrels: list[dict[str, object]] = []
    skip_qa_ids = set(skip_qa_ids or ())
    qa_rows = read_jsonl(qa_path)
    iterator = qa_rows
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(qa_rows, desc="Building weak qrels", unit="QA")
        except ImportError:  # pragma: no cover - optional UI dependency
            pass
    for qa in iterator:
        if str(qa["id"]) in skip_qa_ids:
            continue
        if not qa.get("is_possible", True):
            continue
        article_ids = article_ids_from_qa(qa)
        candidates = [
            chunk
            for article_id in article_ids
            for chunk in chunks_by_article.get(article_id, [])
        ]
        answers = [str(value) for value in qa.get("answers", []) if str(value).strip()]
        if not candidates or not answers:
            continue
        answer = " ".join(answers)
        lexical_scores = [lexical_overlap(answer, str(chunk.get("chunk_text") or chunk.get("text") or "")) for chunk in candidates]
        semantic_scores = [0.0] * len(candidates)
        if semantic_encoder is not None:
            import numpy as np

            vectors = np.asarray(
                semantic_encoder.encode([answer] + [str(chunk.get("chunk_text") or chunk.get("text") or "") for chunk in candidates], normalize_embeddings=True),
                dtype=np.float32,
            )
            semantic_scores = [float(np.dot(vectors[0], vector)) for vector in vectors[1:]]
        scores = [0.5 * lexical + 0.5 * semantic for lexical, semantic in zip(lexical_scores, semantic_scores)] if semantic_encoder else lexical_scores
        best = max(scores)
        if best > 0:
            relevant = [str(chunk["chunk_id"]) for chunk, score in zip(candidates, scores) if score >= best - relevance_tolerance]
        else:
            # Do not make every chunk relevant when an answer cannot be matched.
            relevant = [str(min(candidates, key=lambda chunk: str(chunk["chunk_id"]))["chunk_id"])]
        qrels.append({
            "qa_id": str(qa["id"]), "question": str(qa["question"]), "article_id": str(qa["article_id"]),
            "source_article_ids": article_ids,
            "relevant_chunk_ids": relevant, "label_type": "weak_answer_chunk_match",
            "matching": "lexical_semantic" if semantic_encoder else "lexical", "best_score": round(float(best), 6),
        })
    return qrels


def write_weak_qrels(output_path: str | Path, qrels: list[dict[str, object]]) -> None:
    write_jsonl(output_path, qrels)
