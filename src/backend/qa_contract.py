"""Small, claim-aware offline contract checks for selected QA regressions."""

from __future__ import annotations

from typing import Any, Mapping


QA_CONTRACTS: dict[str, dict[str, Any]] = {
    "211640_1": {
        "answerable": True,
        "required_claims": ("gan", "thận", "lòng", "dạ dày"),
    },
    "211640_3": {
        "answerable": True,
        "required_claims": ("purin", "nước dùng", "axit uric", "thận"),
    },
    "cross_2_1": {
        "answerable": True,
        "required_sources": ("79595", "75539", "78693"),
        "required_claims": ("Cái Bè", "du lịch", "ẩm thực"),
    },
    "152685_5": {"answerable": False},
}


def evaluate_payload(case_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Check answer/refusal and grounded claim/source contract, not lexical fluency."""
    contract = QA_CONTRACTS[case_id]
    answer = str(payload.get("answer") or "")
    status = str(payload.get("answer_status") or "").casefold()
    decision = payload.get("generation_decision")
    decision = decision if isinstance(decision, Mapping) else {}
    route = payload.get("route_decision")
    route = route if isinstance(route, Mapping) else {}
    if not contract["answerable"]:
        passed = status == "abstained" or str(decision.get("decision") or "").upper() == "REFUSE"
        return {"id": case_id, "passed": passed, "reason": "refusal_required"}

    normalized = answer.casefold()
    missing_claims = [claim for claim in contract.get("required_claims", ()) if claim.casefold() not in normalized]
    selected = {str(item).strip() for item in route.get("selected_article_ids", ())}
    citations = payload.get("citations") or []
    cited = {str(item.get("article_id")).strip() for item in citations if isinstance(item, Mapping)}
    sources = selected | cited
    missing_sources = [source for source in contract.get("required_sources", ()) if source not in sources]
    passed = status in {"generated", "extractive_fallback"} and not missing_claims and not missing_sources
    return {
        "id": case_id,
        "passed": passed,
        "reason": "ok" if passed else "grounding_contract_failed",
        "missing_claims": missing_claims,
        "missing_sources": missing_sources,
    }


def evaluate_cases(predictions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate selected regression payloads keyed by dataset QA id."""
    details = [evaluate_payload(case_id, predictions[case_id]) for case_id in QA_CONTRACTS if case_id in predictions]
    return {
        "count": len(details),
        "passed": sum(bool(item["passed"]) for item in details),
        "failed": sum(not item["passed"] for item in details),
        "details": details,
    }
