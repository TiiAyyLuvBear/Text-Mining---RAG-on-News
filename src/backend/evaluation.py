"""Dependency-free lexical RAG telemetry; never factual correctness."""

from __future__ import annotations

import re
import unicodedata
from typing import Any
from .source_identity import source_key

_WORD_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]|$)")
_STOPWORDS = {"và", "là", "có", "cho", "của", "các", "một", "những", "trong", "khi", "được", "với", "này", "đó", "nào", "như", "từ", "về", "theo", "tại", "sau", "trên", "đến", "hay", "thì", "ra", "bao", "nhiêu"}
POLARITY_TIE_TOLERANCE = 0.15  # named/versioned heuristic; widen only after validation
EVALUATION_VERSION = "lexical-v7"


def _tokens(value: str) -> set[str]:
    value = unicodedata.normalize("NFC", str(value or "")).lower()
    return {word for word in _WORD_RE.findall(value) if word not in _STOPWORDS and len(word) > 1}


def context_relevance(question: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
    query = _tokens(question)
    scores = []
    for context in contexts:
        text = " ".join(str(context.get(key, ""))[:4000] for key in ("title", "description", "text"))
        scores.append(round(len(query & _tokens(text)) / len(query), 4) if query else 0.0)
    return {"per_context": scores, "mean": round(sum(scores) / len(scores), 4) if scores else 0.0, "top": round(max(scores), 4) if scores else 0.0, "method": "question/context lexical overlap; diagnostic, not accuracy"}


def _source_id(item: dict[str, Any], index: int) -> str:
    return source_key(item, index)


def source_diversity(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    ids = {_source_id(item, index) for index, item in enumerate(contexts)}
    pool_sources = max((int(item.get("_pool_source_count", 0)) for item in contexts), default=0)
    pool_contexts = max((int(item.get("_pool_context_count", 0)) for item in contexts), default=0)
    return {"contexts": pool_contexts or len(contexts), "unique_sources": pool_sources or len(ids), "unique_articles": pool_sources or len(ids), "unique_articles_deprecated": pool_sources or len(ids), "ratio": round((pool_sources or len(ids)) / (pool_contexts or len(contexts)), 4) if contexts else 0.0, "key": "article_id|url|chunk_id|anonymous(index+sha1(text))"}


def _segments(text: str) -> list[str]:
    segments = []
    for line in str(text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if line:
            segments.extend(part.strip() for part in _SENTENCE_RE.findall(line) if _tokens(part))
    return segments

def _claims(answer: str) -> list[str]:
    claims: list[str] = []
    for line in _segments(answer):
        claims.append(line)
    pending = ""
    merged: list[str] = []
    for claim in claims:
        marker_only = re.fullmatch(r"\s*\[Nguồn\s*\d+\]\s*", claim, flags=re.IGNORECASE)
        if marker_only and merged:
            merged[-1] = (merged[-1] + " " + claim).strip()
        elif marker_only:
            pending = (pending + " " + claim).strip()
        else:
            merged.append((pending + " " + claim).strip() if pending else claim)
            pending = ""
    if pending:
        merged.append(pending)
    return merged


def _negated(value: str) -> bool:
    return bool(re.search(r"(?i)\b(không|chưa|không phải|chẳng|không gây)\b", value))


def _evidence(claim: str, text: str) -> tuple[float, set[bool]]:
    tokens = _tokens(claim)
    windows = [(len(tokens & _tokens(segment)) / len(tokens) if tokens else 0.0, _negated(segment)) for segment in _segments(str(text or "")[:12000])]
    best = max((score for score, _ in windows), default=0.0)
    return best, {polarity for score, polarity in windows if score >= max(0.35, best - POLARITY_TIE_TOLERANCE)}


def claim_support(answer: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    citation_errors: list[dict[str, Any]] = []
    for claim_index, claim in enumerate(_claims(answer)):
        markers = [int(item) for item in re.findall(r"\[Nguồn\s*(\d+)\]", claim, flags=re.IGNORECASE)]
        valid = [item for item in markers if 1 <= item <= len(contexts)]
        if any(item < 1 or item > len(contexts) for item in markers):
            citation_errors.append({"claim_index": claim_index, "type": "out_of_range"})
        orphan = bool(re.fullmatch(r"\s*\[Nguồn\s*\d+\]\s*", claim, flags=re.IGNORECASE))
        if orphan:
            citation_errors.append({"claim_index": claim_index, "type": "orphan"})
        elif not markers and re.search(r"(?i)(theo|nguồn|bài báo|source)", claim):
            citation_errors.append({"claim_index": claim_index, "type": "detached"})
        evidence = [_evidence(claim, item.get("text", "")) for item in contexts]
        opposing = [index + 1 for index, (score, polarities) in enumerate(evidence) if score >= 0.35 and any(polarity != _negated(claim) for polarity in polarities)]
        supporting = [index + 1 for index, (score, polarities) in enumerate(evidence) if score >= 0.35 and _negated(claim) in polarities]
        cited_support = [index for index in valid if index in supporting and len(evidence[index - 1][1]) == 1]
        if supporting and opposing:
            status = "conflicting"
        elif cited_support or (supporting and not opposing):
            status = "supported"
        elif valid and any(index in opposing for index in valid):
            status = "contradicted"
        elif opposing and not supporting:
            status = "contradicted"
        else:
            status = "unknown"
        details.append({"claim_index": claim_index, "status": status, "support_score": round(max((item[0] for item in evidence), default=0.0), 4), "cited_sources": valid, "citation_presence": bool(markers), "citation_index_validity": bool(markers) and not orphan and len(valid) == len(markers), "citation_validity": bool(markers) and not orphan and len(valid) == len(markers), "citation_supported_sources": cited_support, "citation_support": bool(cited_support)})
    total = len(details)
    supported = sum(item["status"] == "supported" for item in details)
    return {"claims": details, "claim_count": total, "supported_claims": supported, "contradicted_claims": sum(item["status"] == "contradicted" for item in details), "conflicting_claims": sum(item["status"] == "conflicting" for item in details), "unknown_claims": sum(item["status"] == "unknown" for item in details), "lexical_support_coverage": round(supported / total, 4) if total else 0.0, "citation_presence": round(sum(item["citation_presence"] for item in details) / total, 4) if total else 0.0, "citation_index_validity": round(sum(item["citation_index_validity"] for item in details) / total, 4) if total else 0.0, "citation_support": round(sum(item["citation_support"] for item in details) / total, 4) if total else 0.0, "citation_errors": citation_errors, "method": "sentence-window lexical diagnostic threshold=0.35; polarity conservative; not factual correctness"}


def evaluate_response(question: str, answer: str, contexts: list[dict[str, Any]], evidence_sufficient: bool) -> dict[str, Any]:
    relevance = context_relevance(question, contexts)
    diversity = source_diversity(contexts)
    support = claim_support(answer, contexts)
    unsupported = support["unknown_claims"] + support["conflicting_claims"] + support["contradicted_claims"]
    expected_citations = support["claim_count"] > 0
    recommend = (not evidence_sufficient or not contexts or support["claim_count"] == 0 or support["contradicted_claims"] > 0 or support["conflicting_claims"] > 0 or relevance["top"] < 0.15 or (support["claim_count"] and support["lexical_support_coverage"] < 0.5) or (expected_citations and (support["citation_index_validity"] < 1.0 or support["citation_support"] < 0.5)))
    return {"evaluation_version": EVALUATION_VERSION, "context_relevance": relevance, "source_diversity": diversity, "claim_support": support, "unsupported_claims": unsupported, "contradiction_detected": bool(support["contradicted_claims"] or support["conflicting_claims"]), "abstention_recommended": recommend, "confidence_semantics": "not calibrated; lexical retrieval/support diagnostics only"}
