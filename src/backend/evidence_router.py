"""Evidence-coverage routing shared by the adaptive news QA pipeline.

This module deliberately makes no routing decision from query type, entity
count, or estimated source count.  Those are planner hints; candidate support
is the sole input to the final route.
"""

from __future__ import annotations

import re
import unicodedata
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
_SUBJECT_NOISE = {"truong", "dh", "dai", "hoc", "cong", "ty", "tap", "doan"}
_LOCAL_NOISE = {
    "ai", "bao", "biet", "cac", "co", "cua", "duoc", "gi", "khi", "la",
    "nao", "nam", "ngay", "nhung", "ra", "sao", "so", "thang", "the",
    "theo", "trong", "tai", "va", "ve",
}
_ENTITY_NOISE = {
    "bao", "bo", "cong", "congty", "cuoc", "du", "duan", "dot", "gia",
    "giay", "hoa", "hoahau", "kiem", "khu", "muc", "ngoai", "nghe", "nghesi",
    "tap", "tinh", "tp", "truong", "vai",
}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _tokens(value: Any) -> set[str]:
    value = _canonical_text(value)
    return {
        token.casefold() for token in re.findall(r"[\wÀ-ỹ]+", value)
        if len(token) > 1
    }


def _literal_tokens(value: Any) -> set[str]:
    """Tokens used for exact anchors, including meaningful one-character values."""
    return {token.casefold() for token in re.findall(r"[\wÀ-ỹ]+", _canonical_text(value))}


def _fold(value: Any) -> str:
    normalized = unicodedata.normalize("NFKD", _text(value).casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char)).replace("đ", "d")


def _canonical_text(value: Any) -> str:
    """Normalize controlled Vietnamese organisation aliases and punctuation.

    This deliberately is not fuzzy matching.  Only known equivalent spellings
    are collapsed, so an unrelated organisation cannot match merely because
    its name looks similar.
    """
    text = _fold(value).replace("’", "'")
    replacements = (
        (r"\b(?:bo\s+)?(?:gd\s*[-/.]?\s*dt|giao\s+duc\s+(?:va\s+)?dao\s+tao)\b", " gddt "),
        (r"\b(?:thanh\s+pho|tp)\s*[.]?\s*ho\s+chi\s+minh\b|\btp[.]?\s*hcm\b", " tphcm "),
        (r"\b(?:dai\s+hoc|dh)\b", " dh "),
        (r"\b(?:nghe\s+si\s+uu\s+tu|nsut)\b", " nsut "),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _hard_anchors(value: Any) -> set[str]:
    """Return only literal values whose substitution changes the answer.

    Planner proper-name hints are deliberately excluded: semantic evidence can
    spell an organisation differently.  Years/figures and symbolic A/B-style
    subjects remain strict because confusing those changes the requested fact.
    """
    # A number embedded in an identifier (C180, Mazda3, 3e) is not a separate
    # numeric anchor.  Standalone symbolic subjects exclude apostrophe-bound
    # name fragments such as the H in H'Hen.
    pattern = r"(?<![\w])\d+(?:[.,]\d+)*(?![\w])|(?<![\w'’])[A-ZĐ](?![\w'’])"
    return {token.casefold() for token in re.findall(pattern, _text(value))}


def _identifier_anchors(value: Any) -> set[str]:
    return {
        re.sub(r"[^a-z0-9]", "", token)
        for token in re.findall(r"(?i)\b(?=[a-z0-9-]*[a-z])(?=[a-z0-9-]*\d)[a-z0-9-]+\b", _fold(value))
        if not re.fullmatch(r"(?:19|20)\d{2}", token)
    }


def _controlled_entity_score(
    question: str,
    evidence_text: str,
    required_entities: Sequence[str] = (),
) -> float:
    query = _canonical_text(question)
    evidence = _canonical_text(evidence_text)
    controlled = [token for token in ("gddt", "tphcm", "nsut") if token in query.split()]
    identifiers = _identifier_anchors(question)
    checks = [token in evidence.split() for token in controlled]
    compact_evidence = re.sub(r"[^a-z0-9]", "", evidence)
    checks.extend(identifier in compact_evidence for identifier in identifiers)
    controlled_score = sum(checks) / len(checks) if checks else 1.0
    evidence_tokens = _tokens(evidence_text)
    entity_scores = []
    for entity in required_entities:
        terms = _tokens(entity) - _ENTITY_NOISE
        if terms:
            entity_scores.append(len(terms & evidence_tokens) / len(terms))
    return min([controlled_score, *entity_scores], default=1.0)


def _has_comparative_conclusion(value: Any) -> bool:
    text = _text(value).casefold()
    return bool(re.search(
        r"\b(?:so sánh|xếp hạng|nghiêm trọng hơn|quan trọng hơn|"
        r"nghiêm trọng nhất|quan trọng nhất|tốt nhất|xấu nhất)\b",
        text,
    ))


def _relation_requirement(value: Any) -> str:
    question = _canonical_text(value)
    if "gia" in question.split() and "bao nhieu" in question:
        return "price"
    if "bao nhieu" in question:
        return "quantity"
    if re.search(r"\b(?:cam nhan|nhan xet)\b", question):
        return "reported_view"
    if "xu phat" in question:
        return "penalty"
    if re.search(r"\b(?:tang truong|doanh thu)\b.+\bnhu the nao\b", question):
        return "revenue_change" if "doanh thu" in question else "change"
    if re.search(r"\bgay ra\s+thiet hai\b|\bthiet hai cu the\b", question):
        return "actual_damage"
    if "ke hoach" in question and re.search(r"\b(?:gi|nao|nhu the nao)\b", question):
        return "plan"
    if "chu nhan" in question and " ai" in f" {question}":
        return "ownership"
    return ""


def _supports_relation(
    requirement: str,
    unit: str,
    required_entities: Sequence[str] = (),
    question: str = "",
) -> bool:
    if not requirement:
        return True
    value = _canonical_text(unit)
    if requirement == "price":
        return bool(re.search(
            r"(?<![\w])\d+(?:[.,]\d+)*(?:\s*)(?:dong|trieu|ty|usd|euro|yen)(?!\w)",
            value,
        ))
    if requirement == "quantity":
        return bool(
            re.search(r"(?<![\w])\d+(?:[.,]\d+)*(?![\w])", value)
            or re.search(r"\b(?:mot|hai|ba|bon|nam|sau|bay|tam|chin|muoi)\b", value)
        )
    if requirement == "reported_view":
        marker_pattern = r"(?:cam thay|cam nhan|nhan xet|chia se|cho biet|danh gia|noi rang|khang dinh)"
        entity_sets = [_tokens(entity) - _ENTITY_NOISE for entity in required_entities]
        entity_sets = [terms for terms in entity_sets if terms]
        unit_tokens = _tokens(unit)
        if any(len(terms & unit_tokens) / len(terms) < 0.5 for terms in entity_sets):
            return False
        if not entity_sets:
            return bool(re.search(rf"\b{marker_pattern}\b", value))
        primary = r"\s+".join(re.escape(token) for token in _canonical_text(required_entities[0]).split())
        return bool(
            re.search(rf"\b{primary}\b(?:\W+\w+){{0,3}}?\W+\b{marker_pattern}\b", value)
            or re.search(rf"\btheo\s+{primary}\b", value)
        )
    if requirement == "penalty":
        marker = bool(re.search(r"\b(?:xu phat|phat tien|muc phat|che tai)\b", value))
        if "thue khoan" in _canonical_text(question):
            return marker and "thue khoan" in value
        return marker
    if requirement == "change":
        return bool(re.search(r"\b(?:tang|giam|tang truong|sut giam|chuyen bien)\b", value))
    if requirement == "revenue_change":
        target_present = "doanh thu" in value
        if re.search(r"\bsau\s+(?:cac\s+)?chuong trinh\b", _canonical_text(question)):
            target_present = target_present and bool(re.search(
                r"\b(?:chuong trinh|truyen thong|dao tao)\b", value,
            ))
        return target_present and bool(re.search(
            r"\b(?:tang|giam|tang truong|sut giam|chuyen bien)\b", value,
        ))
    if requirement == "actual_damage":
        return bool(re.search(
            r"\b(?:thiet hai|hu hong|sap do|do sap|chet|mat tich|cuon troi|tan pha)\b",
            value,
        )) and not bool(re.search(
            r"\b(?:nguy co|du kien|san sang|de phong|neu co|se|han che|nham tranh|phong tranh)\b",
            value,
        ))
    if requirement == "plan":
        return bool(re.search(r"\b(?:ke hoach|du kien|se|muc tieu|dinh huong)\b", value))
    if requirement == "ownership":
        return bool(re.search(r"\b(?:chu nhan|chu xe|so huu|nguoi ban|cua ong|cua ba)\b", value))
    return True


def _subject_terms(value: Any) -> set[str]:
    match = _SUBJECT_PREFIX_RE.search(_text(value))
    if not match:
        return set()
    return _tokens(match.group(1)) - _SUBJECT_NOISE


def support_diagnostics(
    sub_question: str,
    candidate: Mapping[str, Any],
    evidence_type: str = "",
    required_concepts: Sequence[str] = (),
    required_entities: Sequence[str] = (),
    required_numbers: Sequence[str] = (),
    required_dates: Sequence[str] = (),
    temporal_constraints: Sequence[str] = (),
) -> dict[str, Any]:
    """Return local evidence support plus explainable component diagnostics."""
    query = _tokens(sub_question)
    evidence_text = " ".join(_text(candidate.get(key)) for key in ("title", "text", "chunk_text"))
    evidence = _tokens(evidence_text)
    if not query or not evidence:
        return {"support_score": 0.0, "lexical_score": 0.0, "entity_score": 0.0,
                "concept_score": 0.0, "temporal_score": 0.0, "relation_score": 0.0,
                "failure_reason": "no_candidate", "matched_unit": ""}
    # Do not make capitalization-only planner entities mandatory vocabulary.
    # Only answer-changing literal values remain hard anchors.
    subject_terms = _subject_terms(sub_question)
    entity_score = _controlled_entity_score(sub_question, evidence_text, required_entities)
    if entity_score < 0.75:
        return {"support_score": 0.0, "lexical_score": 0.0, "entity_score": entity_score,
                "concept_score": 1.0, "temporal_score": 1.0, "relation_score": 1.0,
                "failure_reason": "entity_mismatch", "matched_unit": ""}
    if subject_terms:
        identity_text = _text(candidate.get("title")) or evidence_text
        identity_tokens = _tokens(identity_text)
        entity_score = min(entity_score, len(subject_terms & identity_tokens) / len(subject_terms))
        if entity_score < 0.5:
            return {"support_score": 0.0, "lexical_score": 0.0, "entity_score": entity_score,
                    "concept_score": 1.0, "temporal_score": 1.0, "relation_score": 1.0,
                    "failure_reason": "entity_mismatch", "matched_unit": ""}
        requested_years = set(re.findall(r"\b(?:19|20)\d{2}\b", sub_question))
        title_years = set(re.findall(r"\b(?:19|20)\d{2}\b", _text(candidate.get("title"))))
        if requested_years and title_years and requested_years.isdisjoint(title_years):
            return {"support_score": 0.0, "lexical_score": 0.0, "entity_score": entity_score,
                    "concept_score": 1.0, "temporal_score": 0.0, "relation_score": 1.0,
                    "failure_reason": "temporal_mismatch", "matched_unit": ""}
    concept_scores = [
        len(tokens & evidence) / len(tokens) if (tokens := _tokens(concept)) else 0.0
        for concept in required_concepts
    ]
    concept_score = min(concept_scores, default=1.0)
    if concept_score < 0.5:
        return {"support_score": 0.0, "lexical_score": 0.0, "entity_score": entity_score,
                "concept_score": concept_score, "temporal_score": 1.0, "relation_score": 1.0,
                "failure_reason": "required_concept_missing", "matched_unit": ""}
    relation_score = 1.0
    if evidence_type == "COMPARATIVE_CONCLUSION" and not _has_comparative_conclusion(evidence_text):
        return {"support_score": 0.0, "lexical_score": 0.0, "entity_score": entity_score,
                "concept_score": concept_score, "temporal_score": 1.0, "relation_score": 0.0,
                "failure_reason": "relation_mismatch", "matched_unit": ""}
    sentences = split_vietnamese_sentences(evidence_text)
    title = _text(candidate.get("title"))
    units = [title, *sentences]
    units.extend(
        f"{left} {right}" for left, right in zip(sentences, sentences[1:])
    )
    hard_anchors = _hard_anchors(sub_question)
    identifier_anchors = _identifier_anchors(sub_question)
    relation_requirement = _relation_requirement(sub_question)
    if relation_requirement:
        relation_score = 0.0
    local_content = {
        token for token in query
        if token not in _LOCAL_NOISE and not token.replace(".", "").replace(",", "").isdigit()
    }
    scores: list[tuple[float, str]] = []
    for unit in units:
        unit_tokens = _tokens(unit)
        local_identity = f"{title} {unit}" if title and unit != title else unit
        if _controlled_entity_score(
            sub_question, local_identity, required_entities,
        ) < 0.75:
            continue
        if hard_anchors and not hard_anchors <= _literal_tokens(unit):
            continue
        compact_unit = re.sub(r"[^a-z0-9]", "", _canonical_text(unit))
        if identifier_anchors and not all(anchor in compact_unit for anchor in identifier_anchors):
            continue
        if (hard_anchors or identifier_anchors) and local_content and not (local_content & unit_tokens):
            continue
        if relation_requirement and not _supports_relation(
            relation_requirement, unit, required_entities, sub_question,
        ):
            continue
        if relation_requirement:
            relation_score = 1.0
        if evidence_type == "COMPARATIVE_CONCLUSION" and not _has_comparative_conclusion(unit):
            continue
        scores.append((len(query & unit_tokens) / len(query), unit))
    lexical_score, matched_unit = max(scores, default=(0.0, ""), key=lambda item: item[0])
    requested_times = list(required_dates) or list(temporal_constraints)
    temporal_score = 1.0
    if requested_times:
        requested_years = set(re.findall(r"\b(?:19|20)\d{2}\b", " ".join(requested_times)))
        if requested_years:
            found_years = set(re.findall(r"\b(?:19|20)\d{2}\b", evidence_text))
            temporal_score = len(requested_years & found_years) / len(requested_years)
    # Required entities/numbers are exposed for diagnostics. Only controlled
    # aliases and model identifiers are hard identity gates; arbitrary NER
    # output is intentionally not made mandatory.
    del required_numbers
    failure_reason = (
        "relation_mismatch" if relation_requirement and relation_score == 0.0
        else "" if lexical_score >= 0.34
        else "support_score_below_threshold"
    )
    return {"support_score": lexical_score, "lexical_score": lexical_score,
            "entity_score": entity_score, "concept_score": concept_score,
            "temporal_score": temporal_score, "relation_score": relation_score,
            "failure_reason": failure_reason, "matched_unit": matched_unit}


def lexical_support(
    sub_question: str,
    candidate: Mapping[str, Any],
    evidence_type: str = "",
    required_concepts: Sequence[str] = (),
) -> float:
    """Backward-compatible numeric support API."""
    return float(support_diagnostics(
        sub_question, candidate, evidence_type, required_concepts,
    )["support_score"])


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


_FAILURE_REASON_PRIORITY: dict[str, int] = {
    # Candidates that progressed further down the pipeline take precedence
    "relation_mismatch": 5,
    "support_score_below_threshold": 4,
    "temporal_mismatch": 3,
    "required_concept_missing": 2,
    "entity_mismatch": 1,
    "no_candidate": 0,
}


def _candidate_diagnostic_priority(row: Mapping[str, Any]) -> tuple[float, int, float, float]:
    """Score key for picking the most informative candidate when resolving failures.

    1. support_score: Any candidate with partial positive support comes first.
    2. failure_reason priority: Candidates failing later stages are more informative.
    3. lexical_score: Higher lexical overlap indicates closer context match.
    4. entity_score: Higher entity overlap breaks remaining ties.
    """
    support = float(row.get("support_score", 0.0))
    reason = str(row.get("failure_reason", "") or "")
    priority = _FAILURE_REASON_PRIORITY.get(reason, 0)
    lexical = float(row.get("lexical_score", 0.0))
    entity = float(row.get("entity_score", 0.0))
    return (support, priority, lexical, entity)


def _select_best_candidate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Select the candidate providing the strongest evidence or diagnostic signal."""
    if not rows:
        return None
    best = max(rows, key=_candidate_diagnostic_priority)
    return dict(best)


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
                diagnostics = support_diagnostics(
                    _text(sub_question.get("text")),
                    candidate,
                    _text(sub_question.get("evidence_type")),
                    [str(item) for item in sub_question.get("required_concepts", [])],
                    [str(item) for item in sub_question.get("required_entities", [])],
                    [str(item) for item in sub_question.get("required_numbers", [])],
                    [str(item) for item in sub_question.get("required_dates", [])],
                    [str(item) for item in sub_question.get("temporal_constraints", [])],
                )
                raw_score = diagnostics["support_score"]
            else:
                raw_score = support_scorer(_text(sub_question.get("text")), candidate)
                diagnostics = {
                    "support_score": raw_score, "lexical_score": raw_score,
                    "entity_score": 1.0, "concept_score": 1.0,
                    "temporal_score": 1.0, "relation_score": 1.0,
                    "failure_reason": "", "matched_unit": "",
                }
            score = max(0.0, min(1.0, float(raw_score)))
            supported = score >= support_threshold
            diagnostic_row = dict(diagnostics)
            diagnostic_row.update({"chunk_id": chunk_id, "article_id": article_id,
                                   "support_score": score, "supports": supported})
            if supported:
                diagnostic_row["failure_reason"] = ""
            elif not diagnostic_row.get("failure_reason"):
                diagnostic_row["failure_reason"] = "support_score_below_threshold"
            rows.append(diagnostic_row)
            if supported:
                covered_articles.add(article_id)
        supports.append(covered_articles)
        best = _select_best_candidate(rows)
        failure_reason = ""
        if not covered_articles:
            failure_reason = best.get("failure_reason") if best else "no_candidate"
        matrix.append({
            "sub_question_id": sub_id,
            "candidates": rows,
            "covered": bool(covered_articles),
            "covered_by_articles": sorted(covered_articles),
            "missing_sub_questions": [],
            "best_candidate": best,
            "failure_reason": failure_reason,
        })
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
