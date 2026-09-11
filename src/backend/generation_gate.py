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
from .context_compression import (
    TokenCounter,
    compress_context_by_sentence,
    pack_contexts_with_budget,
)

GeneratorCallback = Callable[[str, list[dict[str, Any]]], str]
RetryCallback = Callable[[str], dict[str, Any]]

VALID_ROUTES = {"SINGLE_DOC", "REQUIRES_MULTI_DOC", "INSUFFICIENT"}


def _as_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


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
    evidence_plan: dict[str, Any],
    route_decision: dict[str, Any],
) -> str:
    """Build a deterministic query focused on missing sub-questions."""
    sub_questions = {
        str(item.get("id")): str(item.get("text") or "")
        for item in evidence_plan.get("sub_questions", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    missing_values = _as_strings(route_decision.get("missing_sub_questions"))
    focused = [sub_questions.get(value, value) for value in missing_values]
    parts = [_clean_retry_text(value) for value in focused if _clean_retry_text(value)]
    if not parts:
        normalized = evidence_plan.get("normalized_question") or question
        parts.append(_clean_retry_text(str(normalized)))

    for field in ("entities", "numbers", "dates", "temporal_constraints"):
        for value in _as_strings(evidence_plan.get(field)):
            if not any(value.casefold() in part.casefold() for part in parts):
                parts.append(value)
    return " ".join(dict.fromkeys(parts)).strip()


def maybe_rewrite_and_retrieve(
    question: str,
    evidence_plan: dict[str, Any],
    route_decision: dict[str, Any],
    retry_callback: RetryCallback | None,
    *,
    retry_count: int = 0,
) -> dict[str, Any]:
    """Request one upstream retry and validate the returned contract."""
    if route_decision.get("route") != "INSUFFICIENT":
        return {"retried": False, "retry_count": retry_count, "reason": "route_is_answerable"}
    if not route_decision.get("retry_allowed", False):
        return {"retried": False, "retry_count": retry_count, "reason": "retry_not_allowed"}
    if retry_count >= 1:
        return {"retried": False, "retry_count": retry_count, "reason": "retry_limit_reached"}
    if retry_callback is None:
        return {"retried": False, "retry_count": retry_count, "reason": "retry_callback_missing"}

    retrieval_query = build_retry_query(question, evidence_plan, route_decision)
    result = retry_callback(retrieval_query)
    required = {"ranked_candidates", "coverage_matrix", "route_decision"}
    if not isinstance(result, dict) or not required <= result.keys():
        return {
            "retried": True,
            "retry_count": retry_count + 1,
            "retrieval_query": retrieval_query,
            "reason": "invalid_retry_result",
        }
    return {
        "retried": True,
        "retry_count": retry_count + 1,
        "retrieval_query": retrieval_query,
        "reason": "retry_completed",
        "retry_result": result,
    }


def _select_route_contexts(
    route_decision: dict[str, Any],
    ranked_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    route = str(route_decision.get("route") or "")
    selected_ids = _as_strings(route_decision.get("selected_article_ids"))
    if route not in {"SINGLE_DOC", "REQUIRES_MULTI_DOC"} or not selected_ids:
        return []
    allowed = set(selected_ids[:1] if route == "SINGLE_DOC" else selected_ids)
    selected = [dict(item) for item in ranked_candidates or [] if str(item.get("article_id") or "") in allowed]
    def score(item: dict[str, Any]) -> float:
        try:
            return float(item.get("rerank_score", float("-inf")))
        except (TypeError, ValueError):
            return float("-inf")

    selected.sort(key=score, reverse=True)
    for index, item in enumerate(selected, start=1):
        item.setdefault("citation_rank", index)
    return selected


def _refusal(
    reason: str,
    missing_evidence: list[str],
    *,
    verification_status: str = "WARNING",
    verification_errors: list[dict[str, Any]] | None = None,
    compression_stats: dict[str, Any] | None = None,
    retry_count: int = 0,
    retry_query: str = "",
) -> dict[str, Any]:
    return {
        "decision": "REFUSE",
        "answer": "",
        "citations": [],
        "refusal_reason": reason,
        "missing_evidence": missing_evidence,
        "verification_status": verification_status,
        "verification_errors": verification_errors or [],
        "compression_stats": compression_stats or {},
        "retry_count": retry_count,
        "retry_query": retry_query,
    }


def generate_or_refuse(
    question: str,
    evidence_plan: dict[str, Any],
    coverage_matrix: Any,
    route_decision: dict[str, Any],
    ranked_candidates: list[dict[str, Any]],
    *,
    generator_callback: GeneratorCallback | None,
    token_counter: TokenCounter | None = None,
    compression_threshold: float | None = None,
    token_budget: int | None = None,
    retry_count: int = 0,
    retry_query: str = "",
) -> dict[str, Any]:
    """Select routed evidence, compress, generate, verify, and gate output."""
    route = str(route_decision.get("route") or "")
    missing = _as_strings(route_decision.get("missing_sub_questions"))
    if route == "INSUFFICIENT":
        return _refusal(
            "evidence_insufficient_after_retry" if retry_count else "evidence_insufficient",
            missing,
            retry_count=retry_count,
            retry_query=retry_query,
        )
    if route not in VALID_ROUTES:
        return _refusal("invalid_route_decision", missing, verification_status="FAIL")

    contexts = _select_route_contexts(route_decision, ranked_candidates)
    if not contexts:
        return _refusal("selected_evidence_not_found", missing, verification_status="FAIL")

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
        route_decision=route_decision,
        coverage_matrix=coverage_matrix,
        token_counter=token_counter,
    )
    compression_stats = {**compression_stats, **packing_stats}
    if not packed:
        return _refusal(
            "context_budget_empty",
            missing,
            verification_status="FAIL",
            compression_stats=compression_stats,
            retry_count=retry_count,
            retry_query=retry_query,
        )
    if generator_callback is None:
        return _refusal(
            "generator_callback_missing",
            [],
            verification_status="FAIL",
            compression_stats=compression_stats,
            retry_count=retry_count,
            retry_query=retry_query,
        )

    try:
        generated_answer = str(generator_callback(question, packed) or "").strip()
    except Exception as exc:
        return _refusal(
            "generation_unavailable",
            [],
            verification_status="FAIL",
            verification_errors=[{"type": "generation_error", "category": type(exc).__name__}],
            compression_stats=compression_stats,
            retry_count=retry_count,
            retry_query=retry_query,
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
        )
    return {
        "decision": "ANSWER",
        "answer": generated_answer,
        "citations": verification["citations"],
        "refusal_reason": "",
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
    evidence_plan: dict[str, Any],
    coverage_matrix: Any,
    route_decision: dict[str, Any],
    ranked_candidates: list[dict[str, Any]],
    retry_callback: RetryCallback | None = None,
    *,
    generator_callback: GeneratorCallback | None = None,
    token_counter: TokenCounter | None = None,
    compression_threshold: float | None = None,
    token_budget: int | None = None,
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
    if retry.get("reason") == "invalid_retry_result":
        return _refusal(
            "invalid_retry_result",
            _as_strings(route_decision.get("missing_sub_questions")),
            verification_status="FAIL",
            retry_count=retry_count,
            retry_query=retry_query,
        )
    if retry.get("retry_result"):
        result = retry["retry_result"]
        ranked_candidates = result["ranked_candidates"]
        coverage_matrix = result["coverage_matrix"]
        route_decision = result["route_decision"]

    return generate_or_refuse(
        question,
        evidence_plan,
        coverage_matrix,
        route_decision,
        ranked_candidates,
        generator_callback=generator_callback,
        token_counter=token_counter,
        compression_threshold=(
            config.CONTEXT_COMPRESSION_THRESHOLD
            if compression_threshold is None else compression_threshold
        ),
        token_budget=config.CONTEXT_TOKEN_BUDGET if token_budget is None else token_budget,
        retry_count=retry_count,
        retry_query=retry_query,
    )
