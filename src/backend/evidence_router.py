"""Evidence-coverage routing shared by the adaptive news QA pipeline.

This module deliberately makes no routing decision from query type, entity
count, or estimated source count.  Those are planner hints; candidate support
is the sole input to the final route.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from .evaluation import split_vietnamese_sentences

SINGLE_DOC = "SINGLE_DOC"
REQUIRES_MULTI_DOC = "REQUIRES_MULTI_DOC"
INSUFFICIENT = "INSUFFICIENT"

_SUBJECT_PREFIX_RE = re.compile(
    r"(?i)\b(?:trường\s+(?:đh|đại\s+học)|đại\s+học|công\s+ty|tập\s+đoàn|"
    r"bệnh\s+viện|ngân\s+hàng)\s+(.+?)(?=\s+năm\s+\d|[,.;?!]|$)"
)
_SUBJECT_NOISE = {"trường", "đh", "đại", "học", "công", "ty", "tập", "đoàn"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _tokens(value: Any) -> set[str]:
    return {
        token.casefold() for token in re.findall(r"[\wÀ-ỹ]+", _text(value))
        if len(token) > 1
    }


def _hard_anchors(value: Any) -> set[str]:
    """Return only literal values whose substitution changes the answer.

    Planner proper-name hints are deliberately excluded: semantic evidence can
    spell an organisation differently.  Years/figures and symbolic A/B-style
    subjects remain strict because confusing those changes the requested fact.
    """
    return {
        token.casefold()
        for token in re.findall(r"\d+(?:[.,]\d+)*|(?<!\w)[A-ZĐ](?!\w)", _text(value))
    }


def _has_comparative_conclusion(value: Any) -> bool:
    text = _text(value).casefold()
    return bool(re.search(
        r"\b(?:so sánh|xếp hạng|nghiêm trọng hơn|quan trọng hơn|"
        r"nghiêm trọng nhất|quan trọng nhất|tốt nhất|xấu nhất)\b",
        text,
    ))


def _subject_terms(value: Any) -> set[str]:
    match = _SUBJECT_PREFIX_RE.search(_text(value))
    if not match:
        return set()
    return _tokens(match.group(1)) - _SUBJECT_NOISE


def lexical_support(
    sub_question: str,
    candidate: Mapping[str, Any],
    evidence_type: str = "",
    required_concepts: Sequence[str] = (),
) -> float:
    """Conservative deterministic support signal for the no-LLM path."""
    query = _tokens(sub_question)
    evidence_text = " ".join(_text(candidate.get(key)) for key in ("title", "text", "chunk_text"))
    evidence = _tokens(evidence_text)
    if not query or not evidence:
        return 0.0
    # Do not make capitalization-only planner entities mandatory vocabulary.
    # Only answer-changing literal values remain hard anchors.
    subject_terms = _subject_terms(sub_question)
    if subject_terms:
        identity_text = _text(candidate.get("title")) or evidence_text
        identity_tokens = _tokens(identity_text)
        if len(subject_terms & identity_tokens) / len(subject_terms) < 0.5:
            return 0.0
        requested_years = set(re.findall(r"\b(?:19|20)\d{2}\b", sub_question))
        title_years = set(re.findall(r"\b(?:19|20)\d{2}\b", _text(candidate.get("title"))))
        if requested_years and title_years and requested_years.isdisjoint(title_years):
            return 0.0
    if evidence_type == "COMPARATIVE_CONCLUSION" and not _has_comparative_conclusion(evidence_text):
        return 0.0
    if required_concepts and any(
        not (concept_tokens := _tokens(concept))
        or len(concept_tokens & evidence) / len(concept_tokens) < 0.5
        for concept in required_concepts
    ):
        return 0.0
    sentences = split_vietnamese_sentences(evidence_text)
    units = [_text(candidate.get("title")), *sentences]
    units.extend(
        f"{left} {right}" for left, right in zip(sentences, sentences[1:])
    )
    hard_anchors = _hard_anchors(sub_question)
    scores = []
    for unit in units:
        unit_tokens = _tokens(unit)
        if hard_anchors and not hard_anchors <= unit_tokens:
            continue
        if evidence_type == "COMPARATIVE_CONCLUSION" and not _has_comparative_conclusion(unit):
            continue
        scores.append(len(query & unit_tokens) / len(query))
    return max(scores, default=0.0)


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
            # Candidate order is the post-rerank/temporal/diversification
            # evidence order.  Preserve the first equally-small cover instead
            # of replacing it by lexicographically smaller article IDs.  The
            # latter made article "117558" displace stronger evidence from
            # article "211640" merely because its identifier sorts first.
            if merged not in states or len(candidate) < len(states[merged]):
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
            if support_scorer is lexical_support:
                raw_score = lexical_support(
                    _text(sub_question.get("text")),
                    candidate,
                    _text(sub_question.get("evidence_type")),
                    [str(item) for item in sub_question.get("required_concepts", [])],
                )
            else:
                raw_score = support_scorer(_text(sub_question.get("text")), candidate)
            score = max(0.0, min(1.0, float(raw_score)))
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
