"""Deterministic sentence compression and context-budget packing.

This module does not decide whether evidence is sufficient. It preserves the
upstream source/citation identity and only reduces already-selected evidence.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable
from typing import Any

from . import config
from .evaluation import tokenize_text

TokenCounter = Callable[[str], int]

_SPLIT_RE = re.compile(r"(?<=[.!?])(?:\s+|(?=\S))|[\r\n]+|\s*[•●▪◦]+\s*|\s+(?=[-–—]\s+)")


def default_token_count(text: str) -> int:
    """Cheap deterministic fallback when the active model tokenizer is absent."""
    return len(re.findall(r"\w+|[^\w\s]", str(text or ""), re.UNICODE))


def token_counter_from_tokenizer(tokenizer: Any) -> TokenCounter:
    """Adapt the active generator tokenizer without loading another model."""
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TypeError("tokenizer must provide an encode() method")

    def count(text: str) -> int:
        try:
            return len(encode(str(text or ""), add_special_tokens=False))
        except TypeError:
            return len(encode(str(text or "")))

    return count


def sentence_split(context_text: str, **metadata: Any) -> list[dict[str, Any]]:
    """Split Vietnamese text while retaining stable sentence/source metadata."""
    normalized = unicodedata.normalize("NFC", str(context_text or ""))
    normalized = re.sub(r"[\t\f\v]+", " ", normalized)
    normalized = re.sub(r"[ ]{2,}", " ", normalized).strip()
    if not normalized:
        return []

    sentences: list[dict[str, Any]] = []
    for part in _SPLIT_RE.split(normalized):
        text = re.sub(r"^\s*(?:[-–—*]+|\d+[.)])\s*", "", part).strip()
        if not text:
            continue
        sentence = {key: value for key, value in metadata.items() if value is not None}
        sentence.update({"text": text, "sentence_index": len(sentences)})
        sentences.append(sentence)
    return sentences


def _values(plan: dict[str, Any] | None, key: str) -> list[str]:
    value = (plan or {}).get(key, [])
    if isinstance(value, (str, int, float)):
        value = [value]
    return [unicodedata.normalize("NFC", str(item)).lower().strip() for item in value if str(item).strip()]


def score_sentence(
    question: str,
    sentence: str | dict[str, Any],
    evidence_plan: dict[str, Any] | None = None,
) -> float:
    """Score a sentence using lexical coverage plus lightweight exact signals."""
    item = sentence if isinstance(sentence, dict) else {"text": sentence}
    text = unicodedata.normalize("NFC", str(item.get("text") or ""))
    query_tokens = tokenize_text(question)
    sentence_tokens = tokenize_text(text)
    lexical = len(query_tokens & sentence_tokens) / len(query_tokens) if query_tokens else 0.0

    lowered = text.lower()
    signal_values = (
        _values(evidence_plan, "entities")
        + _values(evidence_plan, "numbers")
        + _values(evidence_plan, "dates")
        + _values(evidence_plan, "temporal_constraints")
    )
    matched = sum(1 for value in signal_values if value in lowered)
    signal_bonus = min(0.20, matched * 0.05)

    title_tokens = tokenize_text(str(item.get("title") or ""))
    description_tokens = tokenize_text(str(item.get("description") or ""))
    metadata_bonus = 0.0
    if query_tokens and query_tokens & title_tokens:
        metadata_bonus += 0.05
    if query_tokens and query_tokens & description_tokens:
        metadata_bonus += 0.025
    return round(min(1.0, lexical + signal_bonus + metadata_bonus), 6)


def _coverage_entries(coverage_matrix: Any) -> list[dict[str, Any]]:
    if isinstance(coverage_matrix, dict):
        entries = coverage_matrix.get("coverage") or coverage_matrix.get("items")
        if isinstance(entries, list):
            return [item for item in entries if isinstance(item, dict)]
        return [coverage_matrix] if "sub_question_id" in coverage_matrix else []
    return [item for item in (coverage_matrix or []) if isinstance(item, dict)]


def _candidate_sub_questions(coverage_matrix: Any) -> dict[tuple[str, str], set[str]]:
    mapping: dict[tuple[str, str], set[str]] = {}
    for entry in _coverage_entries(coverage_matrix):
        sub_id = str(entry.get("sub_question_id") or "").strip()
        if not sub_id:
            continue
        for candidate in entry.get("candidates") or []:
            if not isinstance(candidate, dict) or candidate.get("supports") is False:
                continue
            key = (str(candidate.get("article_id") or ""), str(candidate.get("chunk_id") or ""))
            mapping.setdefault(key, set()).add(sub_id)
        for article_id in entry.get("covered_by_articles") or []:
            mapping.setdefault((str(article_id), ""), set()).add(sub_id)
    return mapping


def compress_context_by_sentence(
    question: str,
    contexts: list[dict[str, Any]],
    evidence_plan: dict[str, Any] | None = None,
    coverage_matrix: Any = None,
    *,
    threshold: float | None = None,
    token_counter: TokenCounter | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Prune low-scoring sentences without changing source/citation identity."""
    threshold = config.CONTEXT_COMPRESSION_THRESHOLD if threshold is None else float(threshold)
    counter = token_counter or default_token_count
    coverage = _candidate_sub_questions(coverage_matrix)
    compressed: list[dict[str, Any]] = []
    original_count = kept_count = tokens_before = tokens_after = 0

    for context_index, context in enumerate(contexts or []):
        metadata = {
            key: context.get(key)
            for key in ("article_id", "chunk_id", "citation_rank", "title", "description")
        }
        sentences = sentence_split(str(context.get("text") or ""), **metadata)
        original_count += len(sentences)
        tokens_before += counter(str(context.get("text") or ""))
        article_id = str(context.get("article_id") or "")
        chunk_id = str(context.get("chunk_id") or "")
        covered = coverage.get((article_id, chunk_id), set()) | coverage.get((article_id, ""), set())
        for sentence in sentences:
            sentence["score"] = score_sentence(question, sentence, evidence_plan)
            sentence["covered_sub_questions"] = sorted(covered)
            sentence["context_index"] = context_index

        selected = [sentence for sentence in sentences if sentence["score"] >= threshold]
        if sentences and not selected:
            selected = [max(sentences, key=lambda item: (item["score"], -item["sentence_index"]))]
        selected.sort(key=lambda item: item["sentence_index"])

        item = dict(context)
        item["text"] = " ".join(sentence["text"] for sentence in selected)
        item["_sentences"] = selected
        item["_covered_sub_questions"] = sorted(covered)
        item["original_sentence_count"] = len(sentences)
        item["kept_sentence_count"] = len(selected)
        item["compression_ratio"] = round(len(selected) / len(sentences), 4) if sentences else 0.0
        compressed.append(item)
        kept_count += len(selected)
        tokens_after += counter(item["text"])

    stats = {
        "original_sentence_count": original_count,
        "kept_sentence_count": kept_count,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "compression_ratio": round(tokens_after / tokens_before, 4) if tokens_before else 0.0,
        "threshold": threshold,
    }
    return compressed, stats


def _truncate_to_budget(text: str, budget: int, counter: TokenCounter) -> str:
    if budget <= 0:
        return ""
    words = str(text or "").split()
    if not words:
        return ""
    low, high = 1, len(words)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = " ".join(words[:middle])
        if middle < len(words):
            candidate += "…"
        if counter(candidate) <= budget:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


def _unique_sentences(contexts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for context_index, context in enumerate(contexts):
        sentences = context.get("_sentences")
        if not isinstance(sentences, list):
            sentences = sentence_split(
                str(context.get("text") or ""),
                **{key: context.get(key) for key in ("article_id", "chunk_id", "citation_rank")},
            )
            for sentence in sentences:
                sentence["score"] = float(context.get("rerank_score") or 0.0)
                sentence["covered_sub_questions"] = context.get("_covered_sub_questions", [])
        for sentence in sentences:
            copied = dict(sentence)
            copied["context_index"] = context_index
            result.append(copied)
    return result


def pack_contexts_with_budget(
    contexts: list[dict[str, Any]],
    *,
    token_budget: int | None = None,
    route_decision: dict[str, Any] | None = None,
    coverage_matrix: Any = None,
    token_counter: TokenCounter | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pack sentences in two passes, reserving coverage before score fill."""
    del coverage_matrix  # coverage annotations are attached during compression
    budget = config.CONTEXT_TOKEN_BUDGET if token_budget is None else max(0, int(token_budget))
    counter = token_counter or default_token_count
    sentences = _unique_sentences(contexts or [])
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[int, int]] = set()
    used = 0

    required_articles = [str(value) for value in (route_decision or {}).get("selected_article_ids", [])]
    required_sub_questions = [str(value) for value in (route_decision or {}).get("covered_sub_questions", [])]

    def key(sentence: dict[str, Any]) -> tuple[int, int]:
        return int(sentence.get("context_index", 0)), int(sentence.get("sentence_index", 0))

    def add(sentence: dict[str, Any], *, max_cost: int | None = None) -> bool:
        nonlocal used
        sentence_key = key(sentence)
        if sentence_key in selected_keys:
            return True
        text = str(sentence.get("text") or "")
        cost = counter(text)
        available = budget - used
        allowed = available if max_cost is None else min(available, max(0, max_cost))
        if cost > allowed:
            if allowed > 0 and (max_cost is not None or not selected):
                text = _truncate_to_budget(text, allowed, counter)
                if text:
                    sentence = {**sentence, "text": text, "truncated": True}
                    cost = counter(text)
                else:
                    return False
            else:
                return False
        selected.append(sentence)
        selected_keys.add(sentence_key)
        used += cost
        return True

    # Pass 1a: preserve at least one sentence for every required article.
    for required_index, article_id in enumerate(required_articles):
        options = [item for item in sentences if str(item.get("article_id") or "") == article_id]
        if options:
            remaining_required = len(required_articles) - required_index
            fair_share = (budget - used) // remaining_required if remaining_required else 0
            add(
                max(options, key=lambda item: (float(item.get("score", 0.0)), -int(item.get("sentence_index", 0)))),
                max_cost=fair_share,
            )

    # Pass 1b: preserve one best annotated sentence per covered sub-question.
    for sub_id in required_sub_questions:
        options = [item for item in sentences if sub_id in item.get("covered_sub_questions", [])]
        if options:
            add(max(options, key=lambda item: (float(item.get("score", 0.0)), -int(item.get("sentence_index", 0)))))

    # Pass 2: fill remaining budget globally by relevance.
    for sentence in sorted(
        sentences,
        key=lambda item: (-float(item.get("score", 0.0)), int(item.get("context_index", 0)), int(item.get("sentence_index", 0))),
    ):
        add(sentence)

    selected.sort(key=lambda item: (int(item.get("context_index", 0)), int(item.get("sentence_index", 0))))
    by_context: dict[int, list[dict[str, Any]]] = {}
    for sentence in selected:
        by_context.setdefault(int(sentence.get("context_index", 0)), []).append(sentence)

    packed: list[dict[str, Any]] = []
    for context_index, context in enumerate(contexts or []):
        kept = by_context.get(context_index, [])
        if not kept:
            continue
        item = dict(context)
        item["text"] = " ".join(str(sentence["text"]) for sentence in kept)
        item["_sentences"] = kept
        item["kept_sentence_count"] = len(kept)
        packed.append(item)

    stats = {
        "token_budget": budget,
        "tokens_packed": used,
        "packed_sentence_count": len(selected),
        "packed_context_count": len(packed),
        "required_articles_preserved": [
            article_id for article_id in required_articles
            if any(str(item.get("article_id") or "") == article_id for item in packed)
        ],
        "required_sub_questions_preserved": [
            sub_id for sub_id in required_sub_questions
            if any(sub_id in sentence.get("covered_sub_questions", []) for sentence in selected)
        ],
    }
    return packed, stats
