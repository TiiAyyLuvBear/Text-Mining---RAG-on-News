"""Post-generation lexical claim and citation verification.

The verifier reports generation defects separately from upstream evidence
insufficiency. It is intentionally conservative: lexical ``unknown`` is a
warning, while malformed citations and contradictions are hard failures.
"""

from __future__ import annotations

import re
from typing import Any

from .evaluation import evidence_match, extract_claims, is_negated

_CITATION_RE = re.compile(r"\[Nguồn\s*(\d+)\]", re.IGNORECASE)
_ORPHAN_RE = re.compile(r"^\s*\[Nguồn\s*\d+\]\s*[.!?]?\s*$", re.IGNORECASE)
_REFUSAL_RE = re.compile(r"(?i)^\s*(?:không đủ thông tin|không thể trả lời)")


def _citation_map(contexts: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], set[int]]:
    mapping: dict[int, dict[str, Any]] = {}
    duplicates: set[int] = set()
    for index, context in enumerate(contexts or [], start=1):
        raw_rank = context.get("citation_rank", index)
        try:
            rank = int(raw_rank)
        except (TypeError, ValueError):
            continue
        if rank in mapping:
            duplicates.add(rank)
        else:
            mapping[rank] = context
    return mapping, duplicates


def _plain_claim(claim: str) -> str:
    return _CITATION_RE.sub("", claim).strip(" \t\r\n-–—•.,;")


def claim_and_citation_verifier(
    answer: str,
    contexts: list[dict[str, Any]],
    *,
    support_threshold: float = 0.35,
) -> dict[str, Any]:
    """Verify citation syntax/mapping and lexical support in cited sources."""
    answer = str(answer or "").strip()
    citation_map, duplicate_ranks = _citation_map(contexts)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if not answer:
        errors.append({"type": "empty_answer", "message": "Generator returned no answer."})
    if duplicate_ranks:
        errors.append({
            "type": "duplicate_citation_rank",
            "citation_ranks": sorted(duplicate_ranks),
            "message": "Multiple contexts use the same citation rank.",
        })

    orphan_markers = [
        line.strip() for line in answer.splitlines() if _ORPHAN_RE.fullmatch(line.strip())
    ]
    for marker in orphan_markers:
        errors.append({"type": "orphan_citation", "citation": marker})

    details: list[dict[str, Any]] = []
    for claim_index, claim in enumerate(extract_claims(answer)):
        plain = _plain_claim(claim)
        if not plain or _REFUSAL_RE.match(plain):
            continue
        ranks = [int(value) for value in _CITATION_RE.findall(claim)]
        valid_ranks = [rank for rank in ranks if rank in citation_map and rank not in duplicate_ranks]
        invalid_ranks = [rank for rank in ranks if rank not in citation_map or rank in duplicate_ranks]
        if not ranks:
            errors.append({
                "type": "missing_citation",
                "claim_index": claim_index,
                "claim": plain,
            })
        if invalid_ranks:
            errors.append({
                "type": "invalid_citation",
                "claim_index": claim_index,
                "citation_ranks": invalid_ranks,
            })

        claim_negative = is_negated(plain)
        matches: list[dict[str, Any]] = []
        supporting = opposing = False
        for rank in valid_ranks:
            context = citation_map[rank]
            score, polarities = evidence_match(plain, str(context.get("text") or ""))
            if score >= support_threshold:
                supporting = supporting or claim_negative in polarities
                opposing = opposing or any(polarity != claim_negative for polarity in polarities)
            matches.append({
                "citation_rank": rank,
                "support_score": round(score, 4),
                "article_id": context.get("article_id"),
                "chunk_id": context.get("chunk_id"),
            })

        if supporting and opposing:
            status = "conflicting"
            errors.append({"type": "conflicting_claim", "claim_index": claim_index, "claim": plain})
        elif supporting:
            status = "supported"
        elif opposing:
            status = "contradicted"
            errors.append({"type": "contradicted_claim", "claim_index": claim_index, "claim": plain})
        else:
            status = "unknown"
            if valid_ranks:
                warnings.append({"type": "unknown_claim", "claim_index": claim_index, "claim": plain})

        details.append({
            "claim_index": claim_index,
            "claim": plain,
            "status": status,
            "citation_presence": bool(ranks),
            "citation_validity": bool(ranks) and len(valid_ranks) == len(ranks),
            "cited_sources": valid_ranks,
            "matches": matches,
        })

    cited_ranks = sorted({rank for detail in details for rank in detail["cited_sources"]})
    citations = [
        {
            "citation_rank": rank,
            "article_id": citation_map[rank].get("article_id"),
            "chunk_id": citation_map[rank].get("chunk_id"),
            "title": citation_map[rank].get("title"),
            "url": citation_map[rank].get("url"),
        }
        for rank in cited_ranks
    ]
    status = "FAIL" if errors else "WARNING" if warnings else "PASS"
    return {
        "verification_status": status,
        "verification_errors": errors,
        "verification_warnings": warnings,
        "claims": details,
        "claim_count": len(details),
        "supported_claims": sum(item["status"] == "supported" for item in details),
        "unknown_claims": sum(item["status"] == "unknown" for item in details),
        "contradicted_claims": sum(item["status"] == "contradicted" for item in details),
        "conflicting_claims": sum(item["status"] == "conflicting" for item in details),
        "citations": citations,
        "method": f"cited-source lexical support threshold={support_threshold}; not NLI/factual correctness",
    }
