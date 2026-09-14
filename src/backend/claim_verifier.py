"""Post-generation lexical claim and citation verification.

The verifier reports generation defects separately from upstream evidence
insufficiency. It is intentionally conservative: lexical ``unknown`` is a
warning, while malformed citations and contradictions are hard failures.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from .evaluation import (
    evidence_unit_matches,
    extract_citation_ranks,
    extract_claims,
    is_negated,
    tokenize_text,
)

SUPPORT_THRESHOLD = 0.35
CONTRADICTION_THRESHOLD = 0.65

_CITATION_RE = re.compile(
    r"\[(?:Nguồn\s*)?\d+(?:\s*,\s*(?:Nguồn\s*)?\d+)*\]", re.IGNORECASE
)
_ORPHAN_RE = re.compile(
    r"^\s*\[(?:Nguồn\s*)?\d+(?:\s*,\s*(?:Nguồn\s*)?\d+)*\]\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_REFUSAL_RE = re.compile(r"(?i)^\s*(?:không đủ thông tin|không thể trả lời)")
_MISSING_DATA_META_RE = re.compile(
    r"(?i)^\s*(?:\*{0,2})?(?:phần\s+chưa\s+có\s+dữ\s+liệu(?:\s+trong\s+(?:context|tư\s+liệu))?|"
    r"dữ\s+liệu\s+còn\s+thiếu|thông\s+tin\s+còn\s+thiếu|"
    r"các\s+mục\s+được\s+nêu\s+trong\s+(?:context|tư\s+liệu))\s*:",
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
_DIRECTION_TARGET_BOUNDARY_RE = re.compile(
    r"[.!?;:]|\b(?:nhưng|tuy\s+nhiên|trong\s+khi|và)\b",
    re.IGNORECASE,
)
_DIRECTION_POST_MODIFIERS = {
    "cao", "thấp", "nhanh", "mạnh", "nhẹ", "lên", "xuống", "dần",
    "do", "vì", "bởi", "khi", "nếu", "thì", "rất", "đáng", "kể",
}
_NEGATION_RE = re.compile(r"(?i)\b(?:không|chẳng|chưa)\b")
_NEGATION_SCOPE_BOUNDARY_RE = re.compile(
    r"[.!?;,:]|\b(?:nhưng|tuy\s+nhiên|trong\s+khi|nên|do\s+đó)\b",
    re.IGNORECASE,
)
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


def _ordered_content_tokens(text: str) -> list[str]:
    return [
        token.casefold()
        for token in re.findall(r"[\wÀ-ỹ]+", str(text or ""))
        if tokenize_text(token)
    ]


def _direction_frames(text: str) -> list[dict[str, Any]]:
    """Bind each direction word to its nearest sentence-local target phrase."""
    value = str(text or "")
    frames: list[dict[str, Any]] = []
    for direction, pattern in _DIRECTION_PATTERNS.items():
        for match in pattern.finditer(value):
            before = _DIRECTION_TARGET_BOUNDARY_RE.split(value[:match.start()])[-1]
            after = _DIRECTION_TARGET_BOUNDARY_RE.split(value[match.end():], maxsplit=1)[0]
            before_tokens = _ordered_content_tokens(before)
            after_words = _ordered_content_tokens(after)
            after_tokens = [
                token for token in after_words
                if token not in _DIRECTION_POST_MODIFIERS and token not in {"được", "bị", "làm"}
            ]
            starts_with_modifier = bool(after_words and after_words[0] in _DIRECTION_POST_MODIFIERS)
            target = (
                set(before_tokens[-4:])
                if starts_with_modifier or not after_tokens
                else set(after_tokens[:5])
            )
            frames.append({"direction": direction, "target": target})
    return frames


def _targets_match(left: set[str], right: set[str]) -> bool:
    if not left or not right:
        return False
    overlap = len(left & right)
    return overlap >= min(2, len(left), len(right)) and overlap / min(len(left), len(right)) >= 0.75


def _is_heading(text: str) -> bool:
    """Treat only short colon-terminated labels as headings, not factual lead-ins."""
    value = str(text or "").strip()
    return value.endswith(":") and len(re.findall(r"[\wÀ-ỹ]+", value)) <= 12


def _negation_scopes(text: str) -> list[set[str]]:
    value = str(text or "")
    scopes: list[set[str]] = []
    for match in _NEGATION_RE.finditer(value):
        after = _NEGATION_SCOPE_BOUNDARY_RE.split(value[match.end():], maxsplit=1)[0]
        tokens = set(_ordered_content_tokens(after)[:5])
        if tokens:
            scopes.append(tokens)
    return scopes


def _direct_negation_conflict(left: str, right: str) -> bool:
    """Detect negation only when its local predicate is present in the other text."""
    left_negative = bool(_NEGATION_RE.search(left))
    right_negative = bool(_NEGATION_RE.search(right))
    if left_negative == right_negative:
        return False
    negated, other = (left, right) if left_negative else (right, left)
    other_tokens = set(_ordered_content_tokens(other))
    return any(len(scope & other_tokens) >= min(2, len(scope)) for scope in _negation_scopes(negated))


def _direction_relation(
    claim_frames: list[dict[str, Any]],
    evidence_text: str,
) -> tuple[bool, bool]:
    """Return same-direction and opposite-direction matches for the same target."""
    evidence_frames = _direction_frames(evidence_text)
    same = opposite = False
    for claim_frame in claim_frames:
        for evidence_frame in evidence_frames:
            if not _targets_match(claim_frame["target"], evidence_frame["target"]):
                continue
            if claim_frame["direction"] == evidence_frame["direction"]:
                same = True
            else:
                opposite = True
    return same, opposite


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
        if (
            not plain
            or _REFUSAL_RE.match(plain)
            or _MISSING_DATA_META_RE.match(plain)
            or _is_heading(plain)
        ):
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
        claim_direction_frames = _direction_frames(plain)
        non_direction_claim_tokens = tokenize_text(plain) - {
            token
            for pattern in _DIRECTION_PATTERNS.values()
            for match in pattern.finditer(plain)
            for token in tokenize_text(match.group(0))
        }
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
                number_mismatch = bool(
                    claim_numbers and unit_numbers and claim_numbers.isdisjoint(unit_numbers)
                )
                entity_mismatch = bool(
                    claim_entities and unit_entities and claim_entities.isdisjoint(unit_entities)
                )
                same_direction, opposite_direction = _direction_relation(
                    claim_direction_frames, unit_text,
                ) if claim_directions else (False, False)
                direction_mismatch = bool(claim_directions and opposite_direction)
                direction_unresolved = bool(
                    claim_directions and not same_direction and not opposite_direction
                )
                polarity_mismatch = _direct_negation_conflict(plain, unit_text)
                semantic_mismatch = number_mismatch or entity_mismatch or polarity_mismatch
                numeric_evidence_seen = numeric_evidence_seen or bool(unit_numbers)
                exact_number_seen = exact_number_seen or bool(claim_numbers & unit_numbers)
                if unit_match is unit_matches[0]:
                    best_number_mismatch = number_mismatch
                    best_entity_mismatch = entity_mismatch
                    best_direction_mismatch = direction_mismatch
                relation_score = (
                    len(non_direction_claim_tokens & tokenize_text(unit_text))
                    / len(non_direction_claim_tokens)
                    if non_direction_claim_tokens else 0.0
                )
                contradiction_score = max(unit_score, relation_score) if direction_mismatch else unit_score
                if semantic_mismatch or direction_mismatch:
                    if contradiction_score >= contradiction_threshold:
                        opposing = True
                    else:
                        uncertain_opposition = True
                elif not direction_unresolved:
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
