from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Protocol

from .schema import read_jsonl, write_jsonl
from .tokenize import tokenize_vietnamese


class SemanticEncoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: object) -> object: ...


def lexical_overlap(answer: str, chunk_text: str) -> float:
    answer_terms = set(tokenize_vietnamese(answer))
    chunk_terms = set(tokenize_vietnamese(chunk_text))
    return len(answer_terms & chunk_terms) / len(answer_terms) if answer_terms else 0.0


def build_weak_qrels(
    qa_path: str | Path,
    chunks_path: str | Path,
    *,
    semantic_encoder: SemanticEncoder | None = None,
    relevance_tolerance: float = 0.02,
    show_progress: bool = False,
) -> list[dict[str, object]]:
    """Map answers to their best matching source chunks; labels are explicitly weak."""
    chunks_by_article: dict[str, list[dict[str, object]]] = defaultdict(list)
    for chunk in read_jsonl(chunks_path):
        chunks_by_article[str(chunk["article_id"])].append(chunk)

    qrels: list[dict[str, object]] = []
    qa_rows = read_jsonl(qa_path)
    iterator = qa_rows
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(qa_rows, desc="Building weak qrels", unit="QA")
        except ImportError:  # pragma: no cover - optional UI dependency
            pass
    for qa in iterator:
        if not qa.get("is_possible", True):
            continue
        candidates = chunks_by_article.get(str(qa["article_id"]), [])
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
            "relevant_chunk_ids": relevant, "label_type": "weak_answer_chunk_match",
            "matching": "lexical_semantic" if semantic_encoder else "lexical", "best_score": round(float(best), 6),
        })
    return qrels


def write_weak_qrels(output_path: str | Path, qrels: list[dict[str, object]]) -> None:
    write_jsonl(output_path, qrels)
