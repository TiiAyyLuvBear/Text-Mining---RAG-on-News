"""Post-generation lexical claim and citation verification.

The verifier reports generation defects separately from upstream evidence
insufficiency. It is intentionally conservative: lexical ``unknown`` is a
warning, while malformed citations and contradictions are hard failures.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from .evaluation import evidence_unit_matches, extract_citation_ranks, extract_claims, is_negated

SUPPORT_THRESHOLD = 0.35
CONTRADICTION_THRESHOLD = 0.65

_CITATION_RE = re.compile(r"\[(?:Nguồn\s*)?\d+(?:\s*,\s*\d+)*\]", re.IGNORECASE)
_ORPHAN_RE = re.compile(
    r"^\s*\[(?:Nguồn\s*)?\d+(?:\s*,\s*\d+)*\]\s*[.!?]?\s*$", re.IGNORECASE
)
_REFUSAL_RE = re.compile(r"(?i)^\s*(?:không đủ thông tin|không thể trả lời)")
_MISSING_DATA_META_RE = re.compile(
    r"(?i)^\s*(?:\*{0,2})?(?:phần\s+chưa\s+có\s+dữ\s+liệu(?:\s+trong\s+(?:context|tư\s+liệu))?|"
    r"dữ\s+liệu\s+còn\s+thiếu|thông\s+tin\s+còn\s+thiếu)\s*:",
)
_NUMBER_RE = re.compile(
    r"(?<!\w)(\d+(?:[.,]\d+)*)(?P<scales>(?:\s*(?:nghìn|ngàn|triệu|tỷ)){0,2})(?P<percent>\s*%)?",
    re.IGNORECASE,
)
_ACRONYM_RE = re.compile(r"(?<!\w)[A-ZĐÀ-Ỹ](?:[A-ZĐÀ-Ỹ0-9]*|(?:\.[A-ZĐÀ-Ỹ0-9]+)+)(?!\w)")
_PROPER_NAME_RE = re.compile(r"\b(?:[A-ZĐÀ-Ỹ][a-zà-ỹ]+\s+)+[A-ZĐÀ-Ỹ][a-zà-ỹ]+\b")
_DIRECTION_PATTERNS = {
    "increase": re.compile(r"(?i)\b(?:tăng|gia tăng|tăng trưởng|cao hơn|đi lên)\b"),
    "decrease": re.compile(r"(?i)\b(?:giảm|sụt giảm|suy giảm|thấp hơn|đi xuống)\b"),
}
_SCALES = {"nghìn": Decimal(1000), "ngàn": Decimal(1000), "triệu": Decimal(10**6), "tỷ": Decimal(10**9)}


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


def _numbers(text: str) -> set[tuple[Decimal, bool]]:
    values: set[tuple[Decimal, bool]] = set()
    for match in _NUMBER_RE.finditer(text):
        raw = match.group(1)
        separators = re.findall(r"[.,]", raw)
        parts = re.split(r"[.,]", raw)
        if separators and all(len(part) == 3 for part in parts[1:]):
            normalized = "".join(parts)
        elif separators:
            normalized = "".join(parts[:-1]) + "." + parts[-1]
        else:
            normalized = raw
        try:
            value = Decimal(normalized)
        except InvalidOperation:
            continue
        for scale in re.findall(r"nghìn|ngàn|triệu|tỷ", match.group("scales") or "", re.IGNORECASE):
            value *= _SCALES[scale.casefold()]
        values.add((value.normalize(), bool(match.group("percent"))))
    return values


def _entities(text: str) -> set[str]:
    values = {match.group(0).casefold() for match in _ACRONYM_RE.finditer(text)}
    values.update(match.group(0).casefold() for match in _PROPER_NAME_RE.finditer(text))
    return values


def _directions(text: str) -> set[str]:
    return {name for name, pattern in _DIRECTION_PATTERNS.items() if pattern.search(text)}


def claim_and_citation_verifier(
    answer: str,
    contexts: list[dict[str, Any]],
    *,
    support_threshold: float = SUPPORT_THRESHOLD,
    contradiction_threshold: float = CONTRADICTION_THRESHOLD,
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
        if not plain or _REFUSAL_RE.match(plain) or _MISSING_DATA_META_RE.match(plain):
            continue
        ranks = extract_citation_ranks(claim)
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
        claim_numbers = _numbers(plain)
        claim_entities = _entities(plain)
        claim_directions = _directions(plain)
        matches: list[dict[str, Any]] = []
        supporting = opposing = uncertain_opposition = False
        numeric_evidence_seen = False
        exact_number_seen = False
        for rank in valid_ranks:
            context = citation_map[rank]
            context_text = str(context.get("text") or "")
            unit_matches = evidence_unit_matches(plain, context_text)
            score = max((item["score"] for item in unit_matches), default=0.0)
            best_unit = unit_matches[0]["text"] if unit_matches else ""
            best_number_mismatch = False
            best_entity_mismatch = False
            best_direction_mismatch = False
            for unit_match in unit_matches:
                unit_score = float(unit_match["score"])
                if unit_score < support_threshold:
                    continue
                unit_text = str(unit_match["text"])
                unit_numbers = _numbers(unit_text)
                unit_entities = _entities(unit_text)
                unit_directions = _directions(unit_text)
                number_mismatch = bool(
                    claim_numbers and unit_numbers and claim_numbers.isdisjoint(unit_numbers)
                )
                entity_mismatch = bool(
                    claim_entities and unit_entities and claim_entities.isdisjoint(unit_entities)
                )
                direction_mismatch = bool(
                    ("increase" in claim_directions and "decrease" in unit_directions)
                    or ("decrease" in claim_directions and "increase" in unit_directions)
                )
                polarity_mismatch = bool(unit_match["negative"]) != claim_negative
                semantic_mismatch = (
                    number_mismatch or entity_mismatch or direction_mismatch or polarity_mismatch
                )
                numeric_evidence_seen = numeric_evidence_seen or bool(unit_numbers)
                exact_number_seen = exact_number_seen or bool(claim_numbers & unit_numbers)
                if unit_match is unit_matches[0]:
                    best_number_mismatch = number_mismatch
                    best_entity_mismatch = entity_mismatch
                    best_direction_mismatch = direction_mismatch
                if semantic_mismatch:
                    if unit_score >= contradiction_threshold:
                        opposing = True
                    else:
                        uncertain_opposition = True
                else:
                    supporting = True
            matches.append({
                "citation_rank": rank,
                "support_score": round(score, 4),
                "article_id": context.get("article_id"),
                "chunk_id": context.get("chunk_id"),
                "matched_evidence": best_unit,
                "number_mismatch": best_number_mismatch,
                "entity_mismatch": best_entity_mismatch,
                "direction_mismatch": best_direction_mismatch,
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

        if uncertain_opposition and not opposing:
            warnings.append({
                "type": "uncertain_opposition",
                "claim_index": claim_index,
                "claim": plain,
            })

        if claim_numbers and valid_ranks and not exact_number_seen:
            error_type = "number_mismatch" if numeric_evidence_seen else "unsupported_number"
            errors.append({"type": error_type, "claim_index": claim_index, "claim": plain})

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
        "method": (
            f"sentence/clause-local cited-source lexical support threshold={support_threshold}; "
            f"contradiction threshold={contradiction_threshold}; not NLI/factual correctness"
        ),
    }
