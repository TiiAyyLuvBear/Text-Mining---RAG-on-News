"""Generation-stage boundary for compression, one retry, and verification.

Upstream owns retrieval, evidence analysis, and route decisions. Callers inject
the existing generator and optional retry callback; this module never invokes a
retriever or model implementation directly.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from typing import Any

from . import config
from .claim_verifier import claim_and_citation_verifier
from .contract_adapter import as_dict, as_list, field
from .context_compression import (
    TokenCounter,
    compress_context_by_sentence,
    pack_contexts_with_budget,
)

GeneratorCallback = Callable[[str, list[dict[str, Any]]], str]
RetryCallback = Callable[[str], dict[str, Any]]

VALID_ROUTES = {"SINGLE_DOC", "REQUIRES_MULTI_DOC", "INSUFFICIENT"}


def _as_strings(value: Any) -> list[str]:
    return [str(item).strip() for item in as_list(value) if str(item).strip()]


def _clean_retry_text(value: str) -> str:
    value = unicodedata.normalize("NFC", str(value or ""))
    value = re.sub(r"[?!.]+$", "", value.strip())
    value = re.sub(
        r"(?i)\s+(?:là\s+)?(?:bao nhiêu|gì|nào|như thế nào|ra sao)$",
        "",
        value,
    )
    return re.sub(r"\s+", " ", value).strip()


def build_retry_query(
    question: str,
    evidence_plan: Any,
    route_decision: Any,
) -> str:
    """Build a deterministic query focused on missing sub-questions."""
    sub_questions = {
        str(field(item, "id")): str(field(item, "text", "") or "")
        for item in as_list(field(evidence_plan, "sub_questions", []))
        if field(item, "id") is not None
    }
    missing_values = _as_strings(field(route_decision, "missing_sub_questions", []))
    focused = [
        sub_questions[value] if value in sub_questions else value
        for value in missing_values
        if value in sub_questions or not re.fullmatch(r"sq\d+", value, re.IGNORECASE)
    ]
    parts = [_clean_retry_text(value) for value in focused if _clean_retry_text(value)]
    focused_mode = bool(parts)
    if not parts:
        normalized = field(evidence_plan, "normalized_question", "") or question
        fallback = _clean_retry_text(str(normalized))
        if fallback:
            parts.append(fallback)

    signal_fields = ("dates", "temporal_constraints") if focused_mode else (
        "entities", "numbers", "dates", "temporal_constraints"
    )
    for field_name in signal_fields:
        for value in _as_strings(field(evidence_plan, field_name, [])):
            if not any(value.casefold() in part.casefold() for part in parts):
                parts.append(value)
    unique: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = part.casefold()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(part)
    return " ".join(unique).strip()


def maybe_rewrite_and_retrieve(
    question: str,
    evidence_plan: Any,
    route_decision: Any,
    retry_callback: RetryCallback | None,
    *,
    retry_count: int = 0,
) -> dict[str, Any]:
    """Request one upstream retry and validate the returned contract."""
    if field(route_decision, "route") != "INSUFFICIENT":
        return {"retried": False, "retry_count": retry_count, "reason": "route_is_answerable"}
    if not field(route_decision, "retry_allowed", False):
        return {"retried": False, "retry_count": retry_count, "reason": "retry_not_allowed"}
    if retry_count >= 1:
        return {"retried": False, "retry_count": retry_count, "reason": "retry_limit_reached"}
    if retry_callback is None:
        return {"retried": False, "retry_count": retry_count, "reason": "retry_callback_missing"}

    retrieval_query = build_retry_query(question, evidence_plan, route_decision)
    if not retrieval_query:
        return {"retried": False, "retry_count": retry_count, "reason": "empty_retry_query"}
    try:
        result = retry_callback(retrieval_query)
    except Exception as exc:
        return {
            "retried": True,
            "retry_count": retry_count + 1,
            "retrieval_query": retrieval_query,
            "reason": "retry_callback_error",
            "error_category": type(exc).__name__,
        }
    result = as_dict(result)
    route = as_dict(result.get("route_decision"))
    candidates = result.get("ranked_candidates")
    coverage = result.get("coverage_matrix")
    if (
        not isinstance(candidates, list)
        or coverage is None
        or route.get("route") not in VALID_ROUTES
    ):
        return {
            "retried": True,
            "retry_count": retry_count + 1,
            "retrieval_query": retrieval_query,
            "reason": "invalid_retry_result",
        }
    result["route_decision"] = route
    return {
        "retried": True,
        "retry_count": retry_count + 1,
        "retrieval_query": retrieval_query,
        "reason": "retry_completed",
        "retry_result": result,
    }


def _coverage_by_article(coverage_matrix: Any) -> dict[str, set[str]]:
    entries = field(coverage_matrix, "coverage") or field(coverage_matrix, "items")
    entries = as_list(entries if entries is not None else coverage_matrix)
    mapping: dict[str, set[str]] = {}
    for entry in entries:
        sub_id = str(field(entry, "sub_question_id", "") or "").strip()
        if not sub_id:
            continue
        for candidate in as_list(field(entry, "candidates", [])):
            if field(candidate, "supports", False) is True:
                article_id = str(field(candidate, "article_id", "") or "")
                if article_id:
                    mapping.setdefault(article_id, set()).add(sub_id)
        for article_id in _as_strings(field(entry, "covered_by_articles", [])):
            mapping.setdefault(article_id, set()).add(sub_id)
    return mapping


def _select_route_contexts(
    route_decision: Any,
    ranked_candidates: list[Any],
    coverage_matrix: Any,
) -> tuple[list[dict[str, Any]], str]:
    route = str(field(route_decision, "route", "") or "")
    selected_ids = list(dict.fromkeys(_as_strings(field(route_decision, "selected_article_ids", []))))
    if route not in {"SINGLE_DOC", "REQUIRES_MULTI_DOC"} or not selected_ids:
        return [], "selected_article_ids_missing"
    candidates = [as_dict(item) for item in ranked_candidates or []]
    available = {str(item.get("article_id") or "") for item in candidates}
    missing_articles = [article_id for article_id in selected_ids if article_id not in available]
    if route == "REQUIRES_MULTI_DOC" and missing_articles:
        return [], "selected_articles_not_found"

    def score(item: dict[str, Any]) -> float:
        try:
            return float(item.get("rerank_score", float("-inf")))
        except (TypeError, ValueError):
            return float("-inf")

    coverage = _coverage_by_article(coverage_matrix)
    required_sub_questions = set(_as_strings(field(route_decision, "covered_sub_questions", [])))
    if route == "SINGLE_DOC":
        eligible = [
            article_id for article_id in selected_ids
            if article_id in available and (
                not required_sub_questions or coverage.get(article_id, set()) >= required_sub_questions
            )
        ]
        if not eligible:
            return [], "single_doc_covering_article_not_found"
        chosen = max(
            eligible,
            key=lambda article_id: max(
                (score(item) for item in candidates if str(item.get("article_id") or "") == article_id),
                default=float("-inf"),
            ),
        )
        allowed = {chosen}
    else:
        allowed = set(selected_ids)
        if required_sub_questions and set().union(*(coverage.get(item, set()) for item in allowed)) < required_sub_questions:
            return [], "selected_articles_do_not_cover_route"

    selected = [item for item in candidates if str(item.get("article_id") or "") in allowed]
    selected.sort(key=score, reverse=True)
    used_ranks: set[int] = set()
    for item in selected:
        if item.get("citation_rank") is None:
            continue
        try:
            rank = int(item["citation_rank"])
        except (TypeError, ValueError):
            return [], "invalid_citation_rank"
        if rank in used_ranks:
            return [], "duplicate_citation_rank"
        used_ranks.add(rank)
    next_rank = 1
    for item in selected:
        if item.get("citation_rank") is not None:
            continue
        while next_rank in used_ranks:
            next_rank += 1
        item["citation_rank"] = next_rank
        used_ranks.add(next_rank)
    return selected, ""


def _refusal(
    reason: str,
    missing_evidence: list[str],
    *,
    verification_status: str = "NOT_RUN",
    verification_errors: list[dict[str, Any]] | None = None,
    compression_stats: dict[str, Any] | None = None,
    retry_count: int = 0,
    retry_query: str = "",
    refusal_reason_code: str = "",
    failure_category: str = "",
) -> dict[str, Any]:
    return {
        "decision": "REFUSE",
        "answer": "",
        "citations": [],
        "refusal_reason": reason,
        "refusal_reason_code": refusal_reason_code or reason.upper(),
        "failure_category": failure_category,
        "missing_evidence": missing_evidence,
        "verification_status": verification_status,
        "verification_errors": verification_errors or [],
        "verification_warnings": [],
        "compression_stats": compression_stats or {},
        "retry_count": retry_count,
        "retry_query": retry_query,
    }


def generate_or_refuse(
    question: str,
    evidence_plan: Any,
    coverage_matrix: Any,
    route_decision: Any,
    ranked_candidates: list[Any],
    *,
    generator_callback: GeneratorCallback | None,
    token_counter: TokenCounter | None = None,
    compression_threshold: float | None = None,
    token_budget: int | None = None,
    retry_count: int = 0,
    retry_query: str = "",
) -> dict[str, Any]:
    """Select routed evidence, compress, generate, verify, and gate output."""
    route = str(field(route_decision, "route", "") or "")
    missing = _as_strings(field(route_decision, "missing_sub_questions", []))
    if route == "INSUFFICIENT":
        return _refusal(
            "evidence_insufficient_after_retry" if retry_count else "evidence_insufficient",
            missing,
            retry_count=retry_count,
            retry_query=retry_query,
            refusal_reason_code="RETRY_EXHAUSTED" if retry_count else "INSUFFICIENT_EVIDENCE",
            failure_category="EVIDENCE",
        )
    if route not in VALID_ROUTES:
        return _refusal(
            "invalid_route_decision", missing,
            refusal_reason_code="INVALID_ROUTE_DECISION", failure_category="CONTRACT",
            retry_count=retry_count, retry_query=retry_query,
        )

    contexts, selection_error = _select_route_contexts(route_decision, ranked_candidates, coverage_matrix)
    if not contexts:
        return _refusal(
            selection_error or "selected_evidence_not_found", missing,
            refusal_reason_code="SELECTED_EVIDENCE_INVALID", failure_category="CONTRACT",
            retry_count=retry_count, retry_query=retry_query,
        )
    packing_route = as_dict(route_decision)
    packing_route["selected_article_ids"] = list(dict.fromkeys(
        str(item.get("article_id") or "") for item in contexts if item.get("article_id")
    ))

    compressed, compression_stats = compress_context_by_sentence(
        question,
        contexts,
        evidence_plan,
        coverage_matrix,
        threshold=compression_threshold,
        token_counter=token_counter,
    )
    packed, packing_stats = pack_contexts_with_budget(
        compressed,
        token_budget=token_budget,
        route_decision=packing_route,
        coverage_matrix=coverage_matrix,
        token_counter=token_counter,
    )
    compression_stats = {**compression_stats, **packing_stats}
    if (
        not packing_stats["within_budget"]
        or packing_stats["required_articles_missing"]
        or packing_stats["required_sub_questions_missing"]
    ):
        return _refusal(
            "required_evidence_does_not_fit_context_budget",
            packing_stats["required_sub_questions_missing"],
            compression_stats=compression_stats,
            retry_count=retry_count,
            retry_query=retry_query,
            refusal_reason_code="CONTEXT_BUDGET_INSUFFICIENT",
            failure_category="CONTEXT_BUDGET",
        )
    if not packed:
        return _refusal(
            "context_budget_empty",
            missing,
            compression_stats=compression_stats,
            retry_count=retry_count,
            retry_query=retry_query,
            refusal_reason_code="CONTEXT_BUDGET_EMPTY",
            failure_category="CONTEXT_BUDGET",
        )
    if generator_callback is None:
        return _refusal(
            "generator_callback_missing",
            [],
            compression_stats=compression_stats,
            retry_count=retry_count,
            retry_query=retry_query,
            refusal_reason_code="GENERATOR_MISSING",
            failure_category="CONTRACT",
        )

    try:
        raw_answer = generator_callback(question, packed)
    except Exception as exc:
        return _refusal(
            "generation_unavailable",
            [],
            verification_errors=[{"type": "generation_error", "category": type(exc).__name__}],
            compression_stats=compression_stats,
            retry_count=retry_count,
            retry_query=retry_query,
            refusal_reason_code="GENERATOR_ERROR",
            failure_category="GENERATOR",
        )

    if raw_answer is not None and not isinstance(raw_answer, str):
        return _refusal(
            "generation_response_invalid",
            [],
            verification_errors=[{"type": "invalid_generation_type", "category": type(raw_answer).__name__}],
            compression_stats=compression_stats,
            retry_count=retry_count,
            retry_query=retry_query,
            refusal_reason_code="GENERATOR_RESPONSE_INVALID",
            failure_category="PARSE",
        )
    generated_answer = str(raw_answer or "").strip()

    if not generated_answer:
        return _refusal(
            "generation_empty",
            [],
            compression_stats=compression_stats,
            retry_count=retry_count,
            retry_query=retry_query,
            refusal_reason_code="GENERATOR_EMPTY",
            failure_category="GENERATOR",
        )

    verification = claim_and_citation_verifier(generated_answer, packed)
    if verification["verification_status"] == "FAIL":
        return _refusal(
            "generation_verification_failed",
            [],
            verification_status="FAIL",
            verification_errors=verification["verification_errors"],
            compression_stats=compression_stats,
            retry_count=retry_count,
            retry_query=retry_query,
            refusal_reason_code="VERIFICATION_FAILED",
            failure_category="VERIFICATION",
        )
    return {
        "decision": "ANSWER",
        "answer": generated_answer,
        "citations": verification["citations"],
        "refusal_reason": "",
        "refusal_reason_code": "",
        "failure_category": "",
        "missing_evidence": [],
        "verification_status": verification["verification_status"],
        "verification_errors": verification["verification_errors"],
        "verification_warnings": verification["verification_warnings"],
        "compression_stats": compression_stats,
        "retry_count": retry_count,
        "retry_query": retry_query,
    }


def run_generation_stage(
    question: str,
    evidence_plan: Any,
    coverage_matrix: Any,
    route_decision: Any,
    ranked_candidates: list[Any],
    retry_callback: RetryCallback | None = None,
    *,
    generator_callback: GeneratorCallback | None = None,
    token_counter: TokenCounter | None = None,
    compression_threshold: float | None = None,
    token_budget: int | None = None,
    compression_enabled: bool | None = None,
) -> dict[str, Any]:
    """Single integration boundary owned by the post-routing generation stage."""
    retry = maybe_rewrite_and_retrieve(
        question,
        evidence_plan,
        route_decision,
        retry_callback,
        retry_count=0,
    )
    retry_count = int(retry.get("retry_count", 0))
    retry_query = str(retry.get("retrieval_query") or "")
    if retry.get("reason") in {"invalid_retry_result", "retry_callback_error"}:
        return _refusal(
            str(retry["reason"]),
            _as_strings(field(route_decision, "missing_sub_questions", [])),
            retry_count=retry_count,
            retry_query=retry_query,
            verification_errors=[{
                "type": str(retry["reason"]),
                "category": retry.get("error_category"),
            }],
            refusal_reason_code=(
                "RETRY_CALLBACK_ERROR"
                if retry["reason"] == "retry_callback_error"
                else "INVALID_RETRY_RESULT"
            ),
            failure_category="CONTRACT",
        )
    if retry.get("retry_result"):
        result = retry["retry_result"]
        ranked_candidates = result["ranked_candidates"]
        coverage_matrix = result["coverage_matrix"]
        route_decision = result["route_decision"]

    enabled = config.CONTEXT_COMPRESSION_ENABLED if compression_enabled is None else compression_enabled
    effective_threshold = (
        config.CONTEXT_COMPRESSION_THRESHOLD if compression_threshold is None else compression_threshold
    ) if enabled else 0.0
    decision = generate_or_refuse(
        question,
        evidence_plan,
        coverage_matrix,
        route_decision,
        ranked_candidates,
        generator_callback=generator_callback,
        token_counter=token_counter,
        compression_threshold=effective_threshold,
        token_budget=config.CONTEXT_TOKEN_BUDGET if token_budget is None else token_budget,
        retry_count=retry_count,
        retry_query=retry_query,
    )
    decision.setdefault("compression_stats", {})["compression_enabled"] = bool(enabled)
    return decision
