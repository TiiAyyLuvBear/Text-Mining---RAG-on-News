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
from .contract_adapter import as_list, field
from .evaluation import split_vietnamese_sentences, tokenize_text

TokenCounter = Callable[[str], int]


def format_contexts_for_generation(contexts: list[dict[str, Any]]) -> str:
    """Render contexts exactly as the existing generation prompt expects."""
    blocks = []
    for item in contexts:
        rank = item.get("citation_rank", item.get("rank", 0))
        title = str(item.get("title") or "").strip()
        header = f"[Nguồn {rank}]" + (f" {title}" if title else "")
        blocks.append(f"{header}\n{item.get('text', '')}")
    return "\n\n".join(blocks)

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
    sentences: list[dict[str, Any]] = []
    for part in split_vietnamese_sentences(context_text):
        text = re.sub(r"^\s*(?:[-–—*]+|\d+[.)])\s*", "", part).strip()
        if not text:
            continue
        sentence = {key: value for key, value in metadata.items() if value is not None}
        sentence.update({"text": text, "sentence_index": len(sentences)})
        sentences.append(sentence)
    return sentences


def _values(plan: Any, key: str) -> list[str]:
    return [
        unicodedata.normalize("NFC", str(item)).casefold().strip()
        for item in as_list(field(plan, key, []))
        if str(item).strip()
    ]


def _phrase_present(phrase: str, text: str) -> bool:
    words = [re.escape(word) for word in phrase.split() if word]
    if not words:
        return False
    pattern = r"(?<!\w)" + r"\s+".join(words) + r"(?!\w)"
    return bool(re.search(pattern, text, re.IGNORECASE | re.UNICODE))


def score_sentence(
    question: str,
    sentence: str | dict[str, Any],
    evidence_plan: Any = None,
) -> float:
    """Score a sentence using lexical coverage plus lightweight exact signals."""
    item = sentence if isinstance(sentence, dict) else {"text": sentence}
    text = unicodedata.normalize("NFC", str(item.get("text") or ""))
    query_tokens = tokenize_text(question)
    sentence_tokens = tokenize_text(text)
    lexical = len(query_tokens & sentence_tokens) / len(query_tokens) if query_tokens else 0.0

    lowered = text.casefold()
    signal_values = (
        _values(evidence_plan, "entities")
        + _values(evidence_plan, "numbers")
        + _values(evidence_plan, "dates")
        + _values(evidence_plan, "temporal_constraints")
    )
    matched = sum(1 for value in signal_values if _phrase_present(value, lowered))
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
    entries = field(coverage_matrix, "coverage") or field(coverage_matrix, "items")
    if entries is not None:
        return as_list(entries)
    return as_list(coverage_matrix) if field(coverage_matrix, "sub_question_id") is not None or isinstance(coverage_matrix, (list, tuple)) else []


def _candidate_sub_questions(coverage_matrix: Any) -> dict[tuple[str, str], set[str]]:
    mapping: dict[tuple[str, str], set[str]] = {}
    for entry in _coverage_entries(coverage_matrix):
        sub_id = str(field(entry, "sub_question_id", "") or "").strip()
        if not sub_id:
            continue
        for candidate in as_list(field(entry, "candidates", [])):
            if field(candidate, "supports", False) is not True:
                continue
            key = (str(field(candidate, "article_id", "") or ""), str(field(candidate, "chunk_id", "") or ""))
            mapping.setdefault(key, set()).add(sub_id)
        for article_id in as_list(field(entry, "covered_by_articles", [])):
            mapping.setdefault((str(article_id), ""), set()).add(sub_id)
    return mapping


def compress_context_by_sentence(
    question: str,
    contexts: list[dict[str, Any]],
    evidence_plan: Any = None,
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

    sub_question_text = {
        str(field(item, "id", "")): str(field(item, "text", "") or "")
        for item in as_list(field(evidence_plan, "sub_questions", []))
        if str(field(item, "id", "")).strip()
    }

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
            sentence["covered_sub_questions"] = []
            sentence["context_index"] = context_index

        protected_indices: set[int] = set()
        for sub_id in covered:
            if not sentences:
                continue
            target = sub_question_text.get(sub_id) or question
            best = max(
                sentences,
                key=lambda item: (
                    score_sentence(target, item, evidence_plan),
                    item["score"],
                    -item["sentence_index"],
                ),
            )
            best["covered_sub_questions"].append(sub_id)
            protected_indices.add(best["sentence_index"])

        selected = [
            sentence for sentence in sentences
            if sentence["score"] >= threshold or sentence["sentence_index"] in protected_indices
        ]
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
    seen: dict[tuple[str, str, str], dict[str, Any]] = {}
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
            duplicate_key = (
                str(copied.get("article_id") or ""),
                str(copied.get("chunk_id") or ""),
                re.sub(r"\s+", " ", str(copied.get("text") or "")).casefold().strip(),
            )
            if duplicate_key in seen:
                existing = seen[duplicate_key]
                existing["covered_sub_questions"] = sorted(set(
                    existing.get("covered_sub_questions", [])
                ) | set(copied.get("covered_sub_questions", [])))
                existing["score"] = max(float(existing.get("score", 0.0)), float(copied.get("score", 0.0)))
                continue
            seen[duplicate_key] = copied
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
    del coverage_matrix  # sentence-level proxy annotations are attached during compression
    budget = config.CONTEXT_TOKEN_BUDGET if token_budget is None else max(0, int(token_budget))
    counter = token_counter or default_token_count
    sentences = _unique_sentences(contexts or [])
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[int, int]] = set()
    selected_contexts: set[int] = set()
    used = 0

    required_articles = list(dict.fromkeys(
        str(value) for value in as_list(field(route_decision, "selected_article_ids", []))
        if str(value).strip()
    ))
    required_sub_questions = list(dict.fromkeys(
        str(value) for value in as_list(field(route_decision, "covered_sub_questions", []))
        if str(value).strip()
    ))

    def key(sentence: dict[str, Any]) -> tuple[int, int]:
        return int(sentence.get("context_index", 0)), int(sentence.get("sentence_index", 0))

    def add(sentence: dict[str, Any], *, allow_truncate: bool = False) -> bool:
        nonlocal used
        sentence_key = key(sentence)
        if sentence_key in selected_keys:
            return True
        text = str(sentence.get("text") or "")
        cost = counter(text)
        context_index = int(sentence.get("context_index", 0))
        if context_index not in selected_contexts:
            context = contexts[context_index]
            header = (
                f"[Nguồn {context.get('citation_rank', context.get('rank', 0))}] "
                f"article_id={context.get('article_id')} title={context.get('title')}\n"
            )
            cost += counter(header)
        available = budget - used
        if cost > available:
            if allow_truncate and available > 0:
                header_cost = cost - counter(str(sentence.get("text") or ""))
                text = _truncate_to_budget(text, max(0, available - header_cost), counter)
                if text:
                    sentence = {**sentence, "text": text, "truncated": True}
                    cost = counter(text) + header_cost
                else:
                    return False
            else:
                return False
        selected.append(sentence)
        selected_keys.add(sentence_key)
        selected_contexts.add(context_index)
        used += cost
        return True

    # Pass 1a: select one full proxy-evidence sentence per covered sub-question.
    mandatory: list[dict[str, Any]] = []
    for sub_id in required_sub_questions:
        options = [item for item in sentences if sub_id in item.get("covered_sub_questions", [])]
        if options:
            mandatory.append(max(
                options,
                key=lambda item: (float(item.get("score", 0.0)), -int(item.get("sentence_index", 0))),
            ))

    # Pass 1b: add one full sentence for required articles not represented above.
    represented_articles = {str(item.get("article_id") or "") for item in mandatory}
    for article_id in required_articles:
        if article_id in represented_articles:
            continue
        options = [item for item in sentences if str(item.get("article_id") or "") == article_id]
        if options:
            mandatory.append(max(
                options,
                key=lambda item: (float(item.get("score", 0.0)), -int(item.get("sentence_index", 0))),
            ))
            represented_articles.add(article_id)

    # Required evidence is never truncated: a partial sentence must not be
    # reported as preserved evidence. Missing items are exposed in telemetry.
    mandatory_keys = {key(sentence) for sentence in mandatory}
    for sentence in mandatory:
        add(sentence)

    # Pass 2: fill remaining budget globally by relevance.
    for sentence in sorted(
        sentences,
        key=lambda item: (-float(item.get("score", 0.0)), int(item.get("context_index", 0)), int(item.get("sentence_index", 0))),
    ):
        if key(sentence) not in mandatory_keys:
            add(sentence, allow_truncate=not selected)

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

    preserved_articles = [
        article_id for article_id in required_articles
        if any(str(item.get("article_id") or "") == article_id for item in packed)
    ]
    preserved_sub_questions = [
        sub_id for sub_id in required_sub_questions
        if any(sub_id in sentence.get("covered_sub_questions", []) for sentence in selected)
    ]
    stats = {
        "token_budget": budget,
        "tokens_packed": counter(format_contexts_for_generation(packed)),
        "budget_scope": "rendered context blocks including source headers; prompt instructions/question excluded",
        "within_budget": counter(format_contexts_for_generation(packed)) <= budget,
        "packed_sentence_count": len(selected),
        "packed_context_count": len(packed),
        "required_articles_preserved": preserved_articles,
        "required_articles_missing": [item for item in required_articles if item not in preserved_articles],
        "required_sub_questions_preserved": preserved_sub_questions,
        "required_sub_questions_missing": [
            item for item in required_sub_questions if item not in preserved_sub_questions
        ],
    }
    return packed, stats
