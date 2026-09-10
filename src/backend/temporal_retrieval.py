"""Soft temporal signals for news retrieval."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Mapping

_YEAR = r"(?:19|20)\d{2}"
_MONTH_NAME = (
    r"(?:giêng|một|hai|ba|tư|năm|sáu|bảy|tám|chín|"
    r"mười(?:\s+một|\s+hai)?)"
)
_MONTH = rf"(?:tháng\s*(?:1[0-2]|[1-9])(?:\s+năm\s*{_YEAR})?|tháng\s+{_MONTH_NAME})"
_NUMERIC_DATE = (
    rf"(?:\d{{1,2}}[/.\-]\d{{1,2}}[/.\-]\d{{2,4}}|"
    rf"(?:1[0-2]|0?[1-9])[/.\-]{_YEAR})"
)
_NATURAL_DATE = rf"(?:ngày\s+)?\d{{1,2}}\s+tháng\s+(?:1[0-2]|[1-9])(?:\s+năm\s+{_YEAR})?"
_DATE = rf"(?:{_NUMERIC_DATE}|{_NATURAL_DATE}|{_MONTH})"
_POINT = rf"(?:{_DATE}|(?:năm\s+)?{_YEAR})"
_RANGE = rf"(?:từ\s+)?{_POINT}\s*(?:-|–|—|đến|tới|cho đến)\s*{_POINT}"
_RELATIVE = rf"(?:trước|sau|trong|kể từ)\s+(?:{_POINT}|đó|nay)"
_RELATIVE_EVENT = r"(?:trước khi|sau khi)\s+[^,.;!?]+"
_SEQUENCE = r"(?:trước đó|sau đó|năm nay|năm ngoái|năm sau)"
_TEMPORAL_PATTERN = re.compile(
    rf"(?P<range>{_RANGE})|(?P<relative_event>{_RELATIVE_EVENT})|"
    rf"(?P<relative>{_RELATIVE})|(?P<date>{_DATE})|"
    rf"(?P<year>(?:năm\s+)?{_YEAR})|(?P<sequence>{_SEQUENCE})",
    re.IGNORECASE | re.UNICODE,
)


def _normalize(value: Any) -> str:
    return re.sub(
        r"\s+", " ", unicodedata.normalize("NFC", str(value or ""))
    ).strip().casefold()


def _unique(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def extract_temporal_terms(question: str) -> list[str]:
    """Extract deterministic, non-overlapping temporal expressions."""
    normalized = _normalize(question)
    return _unique(match.group(0) for match in _TEMPORAL_PATTERN.finditer(normalized))


def temporal_terms_from_plan(
    evidence_plan: Any | None = None,
    question: str | None = None,
) -> list[str]:
    """Use planner constraints when available, with raw-question fallback."""
    if evidence_plan is not None:
        if isinstance(evidence_plan, Mapping):
            constraints = evidence_plan.get("temporal_constraints") or evidence_plan.get("dates")
            normalized_question = evidence_plan.get("normalized_question")
        else:
            constraints = getattr(evidence_plan, "temporal_constraints", None) or getattr(
                evidence_plan, "dates", None
            )
            normalized_question = getattr(evidence_plan, "normalized_question", None)
        planned = _unique(constraints or [])
        if planned:
            return planned
        question = question or normalized_question
    return extract_temporal_terms(question or "")


def _as_dict(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, Mapping):
        return dict(candidate)
    if hasattr(candidate, "model_dump"):
        return candidate.model_dump(by_alias=True)
    raise TypeError("candidate must be a mapping or Pydantic model")


def _candidate_text(candidate: dict[str, Any]) -> str:
    return _normalize(
        " ".join(
            str(candidate.get(field, ""))
            for field in (
                "title",
                "text",
                "chunk_text",
                "description",
                "date",
                "published_at",
                "published_date",
            )
        )
    )


def _base_score(candidate: dict[str, Any]) -> float:
    for field in ("rerank_score", "score", "retrieval_score"):
        try:
            if candidate.get(field) is not None:
                return float(candidate[field])
        except (TypeError, ValueError):
            pass
    return 0.0


def apply_temporal_boost(
    candidates: Iterable[Any],
    terms: Iterable[str],
    *,
    boost: float = 0.05,
) -> list[dict[str, Any]]:
    """Apply a capped soft boost and preserve stable ordering for score ties."""
    normalized_terms = _unique(terms)
    scored: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        item = _as_dict(candidate)
        matched = [term for term in normalized_terms if term in _candidate_text(item)]
        item["temporal_terms_matched"] = matched
        item["temporal_boost"] = boost if matched else 0.0
        item["temporal_score"] = _base_score(item) + item["temporal_boost"]
        item["_temporal_input_index"] = index
        scored.append(item)
    scored.sort(
        key=lambda item: (
            -float(item["temporal_score"]),
            item["_temporal_input_index"],
        )
    )
    for item in scored:
        item.pop("_temporal_input_index", None)
    return scored
