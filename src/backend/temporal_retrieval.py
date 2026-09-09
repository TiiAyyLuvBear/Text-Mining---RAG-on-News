"""Soft temporal signals for news retrieval."""
from __future__ import annotations
import re
import unicodedata
from typing import Any, Iterable
_YEAR = r"(?:19|20)\d{2}"
_MONTH_NAME = r"(?:gi\u00eang|m\u1ed9t|hai|ba|t\u01b0|n\u0103m|s\u00e1u|b\u1ea3y|t\u00e1m|ch\u00edn|m\u01b0\u1eddi(?:\s+m\u1ed9t|\s+hai)?)"
_MONTH = rf"(?:th\u00e1ng\s*(?:1[0-2]|[1-9])(?:\s+n\u0103m\s*\d{{4}})?|th\u00e1ng\s+{_MONTH_NAME})"
_DATE = rf"(?:\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}|{_MONTH}|\d{{1,2}}\s+th\u00e1ng\s+(?:1[0-2]|[1-9]))"
_RANGE = rf"(?:{_DATE}|{_YEAR})\s*(?:-|\u2013|\u2014|\u0111\u1ebfn|t\u1edbi|cho \u0111\u1ebfn)\s*(?:{_DATE}|{_YEAR})"
_SEQUENCE = r"(?:tr\u01b0\u1edbc \u0111\u00f3|sau \u0111\u00f3|tr\u01b0\u1edbc|sau|k\u1ec3 t\u1eeb|t\u1eeb|\u0111\u1ebfn|t\u1edbi|trong n\u0103m|n\u0103m nay|n\u0103m ngo\u00e1i|n\u0103m sau)"
_TEMPORAL_PATTERN = re.compile(rf"(?P<range>{_RANGE})|(?P<date>{_DATE})|(?P<year>{_YEAR})|(?P<sequence>{_SEQUENCE})", re.IGNORECASE | re.UNICODE)
def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value or ""))).strip().casefold()
def extract_temporal_terms(question: str) -> list[str]:
    normalized = _normalize(question)
    terms=[]; occupied=[]
    for match in _TEMPORAL_PATTERN.finditer(normalized):
        if any(start <= match.start() and match.end() <= end for start,end in occupied): continue
        term=re.sub(r"\s+", " ", match.group(0)).strip()
        if term and term not in terms: terms.append(term)
        occupied.append((match.start(), match.end()))
    for match in re.finditer(rf"(?<!\d){_YEAR}(?!\d)", normalized):
        if match.group(0) not in terms: terms.append(match.group(0))
    return terms
def _candidate_text(candidate: dict[str, Any]) -> str:
    return _normalize(" ".join(str(candidate.get(field, "")) for field in ("title","text","description","date","published_at","published_date")))
def _base_score(candidate: dict[str, Any]) -> float:
    for field in ("rerank_score","score","retrieval_score"):
        try:
            if candidate.get(field) is not None: return float(candidate[field])
        except (TypeError,ValueError): pass
    return 0.0
def apply_temporal_boost(candidates: Iterable[dict[str, Any]], terms: Iterable[str], *, boost: float=0.05) -> list[dict[str, Any]]:
    normalized_terms=[term for term in (_normalize(term) for term in terms) if term]
    scored=[]
    for index,candidate in enumerate(candidates):
        item=dict(candidate); matched=[term for term in normalized_terms if term in _candidate_text(item)]
        item["temporal_terms_matched"]=matched; item["temporal_boost"]=min(boost, boost*len(matched)) if matched else 0.0
        item["temporal_score"]=_base_score(item)+item["temporal_boost"]; item["_temporal_input_index"]=index; scored.append(item)
    scored.sort(key=lambda item:(-float(item["temporal_score"]),item["_temporal_input_index"]))
    for item in scored: item.pop("_temporal_input_index",None)
    return scored
