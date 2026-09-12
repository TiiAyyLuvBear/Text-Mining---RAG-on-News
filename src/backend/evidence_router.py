"""Evidence-coverage routing shared by the adaptive news QA pipeline.

This module deliberately makes no routing decision from query type, entity
count, or estimated source count.  Those are planner hints; candidate support
is the sole input to the final route.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Callable

SINGLE_DOC = "SINGLE_DOC"
REQUIRES_MULTI_DOC = "REQUIRES_MULTI_DOC"
INSUFFICIENT = "INSUFFICIENT"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _tokens(value: Any) -> set[str]:
    return {
        token.casefold() for token in re.findall(r"[\wÀ-ỹ]+", _text(value))
        if len(token) > 1
    }


def lexical_support(sub_question: str, candidate: Mapping[str, Any]) -> float:
    """Conservative deterministic support signal for the no-LLM path."""
    query = _tokens(sub_question)
    evidence_text = " ".join(_text(candidate.get(key)) for key in ("title", "text", "chunk_text"))
    evidence = _tokens(evidence_text)
    if not query or not evidence:
        return 0.0
    # Proper names, dates, and figures are discriminative evidence anchors.
    # Do not let a generic overlap such as "doanh thu năm 2025" claim support
    # for the wrong company or number.
    anchors = {
        value.casefold() for value in re.findall(r"\b(?:[A-ZÀ-Ỹ][\wÀ-ỹ-]*|\d+(?:[.,]\d+)*)\b", _text(sub_question))
        if value.casefold() not in {"ai", "cái", "hãy", "so", "vì", "khi", "tại"}
    }
    if anchors and not anchors <= evidence:
        return 0.0
    return len(query & evidence) / len(query)


def _minimal_cover(supports: list[set[str]], article_order: list[str]) -> list[str] | None:
    required_mask = (1 << len(supports)) - 1
    masks: list[tuple[str, int]] = []
    for article in article_order:
        mask = sum(1 << index for index, articles in enumerate(supports) if article in articles)
        if mask:
            masks.append((article, mask))
    states: dict[int, tuple[str, ...]] = {0: ()}
    for article, mask in masks:
        for prior, chosen in list(states.items()):
            merged = prior | mask
            candidate = chosen + (article,)
            if merged not in states or (len(candidate), candidate) < (len(states[merged]), states[merged]):
                states[merged] = candidate
    result = states.get(required_mask)
    return list(result) if result is not None else None


def route_evidence(
    evidence_plan: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    support_scorer: Callable[[str, Mapping[str, Any]], float] = lexical_support,
    support_threshold: float = 0.34,
) -> dict[str, Any]:
    """Build CoverageMatrix and route strictly from per-subquestion evidence."""
    sub_questions = list(evidence_plan.get("sub_questions") or [])
    article_order: list[str] = []
    supports: list[set[str]] = []
    matrix: list[dict[str, Any]] = []
    for sub_question in sub_questions:
        sub_id = _text(sub_question.get("id"))
        rows = []
        covered_articles: set[str] = set()
        for index, candidate in enumerate(candidates):
            article_id = _text(candidate.get("article_id")) or f"anonymous:{index}"
            chunk_id = _text(candidate.get("chunk_id")) or str(index)
            if article_id not in article_order:
                article_order.append(article_id)
            score = max(0.0, min(1.0, float(support_scorer(_text(sub_question.get("text")), candidate))))
            supported = score >= support_threshold
            rows.append({"chunk_id": chunk_id, "article_id": article_id, "support_score": score, "supports": supported})
            if supported:
                covered_articles.add(article_id)
        supports.append(covered_articles)
        matrix.append({"sub_question_id": sub_id, "candidates": rows, "covered": bool(covered_articles), "covered_by_articles": sorted(covered_articles), "missing_sub_questions": []})
    missing = [
        _text(sub_question.get("id")) for sub_question, articles in zip(sub_questions, supports)
        if not articles
    ]
    for row in matrix:
        row["missing_sub_questions"] = missing
    selected = None if missing else _minimal_cover(supports, article_order)
    if missing or selected is None:
        route, reason, selected = INSUFFICIENT, "missing_evidence", []
    elif len(selected) == 1:
        route, reason = SINGLE_DOC, "one_article_covers_all"
    else:
        route, reason = REQUIRES_MULTI_DOC, "multiple_articles_required"
    return {
        "coverage_matrix": matrix,
        "route_decision": {
            "route": route,
            "reason": reason,
            "covered_sub_questions": [
                _text(item.get("id")) for item in sub_questions if _text(item.get("id")) not in missing
            ],
            "missing_sub_questions": missing,
            "selected_article_ids": selected,
            "retry_allowed": route == INSUFFICIENT,
        },
    }
