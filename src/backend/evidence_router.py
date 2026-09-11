"""Coverage-based evidence planning and answer routing.

Router uses question decomposition plus batched entailment scores.  It does
not inspect dataset labels (in particular, ``qa_type``), so same code works
for production traffic and benchmark fixtures.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence


SINGLE_DOC = "SINGLE_DOC"
REQUIRES_MULTI_DOC = "REQUIRES_MULTI_DOC"
INSUFFICIENT = "INSUFFICIENT"
ANSWER = "ANSWER"
REFUSE = "REFUSE"
EVIDENCE_PLAN_UNAVAILABLE = "evidence_plan_unavailable"
LOGGER = logging.getLogger("rag-api.evidence-router")


@dataclass(frozen=True)
class CoverageCell:
    """One entailment result for one subquestion and one chunk."""

    sub_question_id: str
    article_id: str
    chunk_id: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "article_id": self.article_id,
            "support_score": self.score,
            "supports": False,
        }


@dataclass
class EvidencePlanResult:
    """Serializable public evidence-routing contract."""

    evidence_plan: dict[str, Any]
    coverage_matrix: list[dict[str, Any]] = field(default_factory=list)
    route_decision: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_plan": self.evidence_plan,
            "coverage_matrix": self.coverage_matrix,
            "route_decision": self.route_decision,
        }


def _text(value: Any) -> str:
    return str(value or "").strip()


def _source_id(item: Mapping[str, Any], index: int) -> str:
    article = _text(item.get("article_id"))
    if article:
        return article
    url = _text(item.get("url"))
    if url:
        return url
    return f"anonymous:{index}"


def _chunk_id(item: Mapping[str, Any], index: int) -> str:
    return _text(item.get("chunk_id") or item.get("id")) or str(index)


def _bounded_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return min(1.0, max(0.0, score))


def _words(value: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ỹ]+", value.lower(), flags=re.UNICODE))


def _lexical_support(subquestion: str, text: str) -> float:
    """Small deterministic fallback scorer for local/demo use.

    Real deployments can inject NLI/LLM scorer. Exact phrase and full token
    containment are treated as supported; unrelated text scores zero.
    """
    query = _words(subquestion)
    context = _words(text)
    if not query or not context:
        return 0.0
    overlap = len(query & context) / len(query)
    if subquestion.casefold() in text.casefold():
        return 1.0
    return overlap


def _json_payload(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return value
    if hasattr(value, "content"):
        value = value.content
    if isinstance(value, list) and value and isinstance(value[0], Mapping):
        value = value[0].get("text", value)
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        return json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _invoke(callback: Callable[..., Any], *args: Any) -> Any:
    """Call injected callbacks while allowing common one/two-arg shapes."""
    try:
        return callback(*args)
    except TypeError:
        try:
            return callback(args[0])
        except TypeError:
            # Preserve useful callback errors; caller turns them into refusal.
            raise


class EvidencePlan:
    """Build coverage matrix and route question to answer/refusal.

    ``planner`` may return a list of subquestions or JSON containing
    ``subquestions``. ``entailment`` receives a batch of
    ``{"subquestion", "article_id", "chunk_id", "text"}`` records and may
    return scores, score dictionaries, or records containing ``score``.
    """

    def __init__(
        self,
        planner: Callable[..., Any] | None = None,
        entailment: Callable[..., Any] | None = None,
        *,
        support_threshold: float = 0.70,
        timeout: float | None = None,
        entailment_batch_size: int = 10,
        llm: Callable[..., Any] | None = None,
        entailment_scorer: Callable[..., Any] | None = None,
        support_scorer: Callable[..., Any] | None = None,
    ) -> None:
        # Aliases keep injection ergonomic for tests and provider adapters.
        self.planner = planner or llm
        self.entailment = entailment or entailment_scorer or support_scorer
        self.support_threshold = _bounded_score(support_threshold)
        self.timeout = timeout
        self.entailment_batch_size = max(1, int(entailment_batch_size))

    def _evidence_plan(self, question: str, contexts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if self.planner is None:
            raise ValueError("planner is not configured")
        result = _invoke(self.planner, question, list(contexts))
        result = _json_payload(result)
        # Legacy list support keeps injected test adapters usable.
        if isinstance(result, (list, tuple)):
            result = {"sub_questions": list(result)}
        if not isinstance(result, Mapping):
            raise ValueError("planner output must be an EvidencePlan object")
        raw_questions = result.get("sub_questions", result.get("subquestions"))
        if not isinstance(raw_questions, (list, tuple)):
            raise ValueError("planner output must contain sub_questions")
        sub_questions: list[dict[str, str]] = []
        for index, item in enumerate(raw_questions, start=1):
            if not isinstance(item, Mapping):
                item = {"text": item}
            text = _text(item.get("text") or item.get("subquestion") or item.get("question"))
            if text:
                sub_questions.append({
                    "id": _text(item.get("id")) or f"sq{index}",
                    "text": text,
                    "evidence_type": _text(item.get("evidence_type")).upper() or "FACT",
                })
        if not sub_questions:
            raise ValueError("planner returned no subquestions")
        query_type_hint = _text(result.get("query_type_hint")).upper()
        if query_type_hint not in {"FACTOID", "COMPARISON", "TIMELINE", "GENERAL"}:
            query_type_hint = "GENERAL"
        answer_operator = _text(result.get("answer_operator")).upper()
        if answer_operator not in {"DIRECT", "COMPARE", "TIMELINE", "CAUSAL_SUMMARY"}:
            answer_operator = "DIRECT"
        return {
            "normalized_question": _text(result.get("normalized_question")) or question,
            "query_type_hint": query_type_hint,
            "entities": list(result.get("entities") or []),
            "numbers": list(result.get("numbers") or []),
            "dates": list(result.get("dates") or []),
            "temporal_constraints": list(result.get("temporal_constraints") or []),
            "estimated_sources_needed": max(1, int(result.get("estimated_sources_needed") or 1)),
            "answer_operator": answer_operator,
            "sub_questions": sub_questions,
        }

    @staticmethod
    def _parse_scores(raw: Any, expected_count: int) -> list[float]:
        raw = _json_payload(raw)
        if isinstance(raw, Mapping):
            raw = raw.get("scores", raw.get("results"))
        if not isinstance(raw, (list, tuple)):
            raise ValueError("entailment output must be a score list")
        values: list[float] = []
        for item in raw:
            if isinstance(item, Mapping):
                item = item.get("score", item.get("support", item.get("entailment")))
                if isinstance(item, Mapping):
                    item = item.get("score")
            values.append(_bounded_score(item))
        if len(values) != expected_count:
            raise ValueError("entailment score count does not match batch")
        return values

    def _scores(self, pairs: list[dict[str, Any]]) -> list[float]:
        if self.entailment is None:
            raise ValueError("entailment scorer is not configured")
        values: list[float] = []
        for start in range(0, len(pairs), self.entailment_batch_size):
            batch = pairs[start : start + self.entailment_batch_size]
            batch_started = time.monotonic()
            values.extend(self._parse_scores(_invoke(self.entailment, batch), len(batch)))
            elapsed = time.monotonic() - batch_started
            if self.timeout is not None and elapsed > self.timeout:
                LOGGER.warning("coverage entailment slow | pairs=%d | elapsed_s=%.2f | threshold_s=%.2f", len(batch), elapsed, self.timeout)
        return values

    def _minimal_cover(self, supports: list[set[str]], article_order: list[str]) -> list[str] | None:
        needed = len(supports)
        if not needed:
            return []
        all_mask = (1 << needed) - 1
        article_masks = []
        for article in article_order:
            mask = 0
            for index, covered in enumerate(supports):
                if article in covered:
                    mask |= 1 << index
            if mask:
                article_masks.append((article, mask))
        dp: dict[int, tuple[str, ...]] = {0: ()}
        for article, article_mask in article_masks:
            for mask, chosen in list(dp.items()):
                merged = mask | article_mask
                candidate = chosen + (article,)
                previous = dp.get(merged)
                if previous is None or (len(candidate), candidate) < (len(previous), previous):
                    dp[merged] = candidate
        result = dp.get(all_mask)
        return list(result) if result is not None else None

    def build_evidence_plan(self, question: str) -> dict[str, Any]:
        """Analyze one incoming question before any retrieval occurs."""
        started = time.monotonic()
        result = self._evidence_plan(_text(question), [])
        elapsed = time.monotonic() - started
        if self.timeout is not None and elapsed > self.timeout:
            LOGGER.warning("evidence planner slow | elapsed_s=%.2f | threshold_s=%.2f", elapsed, self.timeout)
        return result

    def plan(
        self,
        question: str,
        contexts: Sequence[Mapping[str, Any]],
        *,
        evidence_plan: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return route decision. Planner/parse failures fail closed."""
        contexts = list(contexts or [])
        started = time.monotonic()
        try:
            evidence_plan = (
                self._evidence_plan(_text(question), contexts)
                if evidence_plan is None
                else self._evidence_plan(_text(question), []) if not evidence_plan else dict(evidence_plan)
            )
            pairs = [
                {
                    "subquestion": sub_question["text"],
                    "article_id": _source_id(context, index),
                    "chunk_id": _chunk_id(context, index),
                    "text": _text(context.get("text")),
                }
                for sub_question in evidence_plan["sub_questions"]
                for index, context in enumerate(contexts)
            ]
            scores = self._scores(pairs)
        except Exception as exc:
            LOGGER.warning("coverage routing unavailable | error=%s", type(exc).__name__)
            return EvidencePlanResult(
                evidence_plan={},
                route_decision={
                    "route": INSUFFICIENT,
                    "reason": EVIDENCE_PLAN_UNAVAILABLE,
                    "covered_sub_questions": [],
                    "missing_sub_questions": [],
                    "selected_article_ids": [],
                },
            ).as_dict()

        elapsed = time.monotonic() - started
        if self.timeout is not None and elapsed > self.timeout:
            LOGGER.warning("coverage routing slow | pairs=%d | elapsed_s=%.2f | threshold_s=%.2f", len(pairs), elapsed, self.timeout)

        by_question: list[dict[str, Any]] = []
        supports: list[set[str]] = []
        article_order = []
        offset = 0
        for sub_question in evidence_plan["sub_questions"]:
            entries = []
            covered_articles: set[str] = set()
            for index, context in enumerate(contexts):
                article = _source_id(context, index)
                if article not in article_order:
                    article_order.append(article)
                score = scores[offset]
                offset += 1
                cell = CoverageCell(sub_question["id"], article, _chunk_id(context, index), score).as_dict()
                cell["supports"] = score >= self.support_threshold
                entries.append(cell)
                if score >= self.support_threshold:
                    covered_articles.add(article)
            supports.append(covered_articles)
            supporting_evidence = [entry for entry in entries if entry["support_score"] >= self.support_threshold]
            by_question.append({
                "sub_question_id": sub_question["id"],
                "candidates": entries,
                "covered": bool(supporting_evidence),
                "covered_by_articles": sorted({entry["article_id"] for entry in supporting_evidence}),
                "missing_sub_questions": [],
            })

        required = self._minimal_cover(supports, article_order)
        missing = [sub_question["id"] for sub_question, covered in zip(evidence_plan["sub_questions"], supports) if not covered]
        for row in by_question:
            row["missing_sub_questions"] = missing
        if missing or required is None:
            route = INSUFFICIENT
            reason = "missing_evidence"
            required = required or []
        else:
            route = SINGLE_DOC if len(required) == 1 else REQUIRES_MULTI_DOC
            reason = "one_article_covers_all" if route == SINGLE_DOC else "multiple_articles_required"
        return EvidencePlanResult(
            evidence_plan=evidence_plan,
            coverage_matrix=by_question,
            route_decision={
                "route": route,
                "reason": reason,
                "covered_sub_questions": [item["id"] for item in evidence_plan["sub_questions"] if item["id"] not in missing],
                "missing_sub_questions": missing,
                "selected_article_ids": required,
            },
        ).as_dict()

    __call__ = plan


def route_evidence(
    question: str,
    contexts: Sequence[Mapping[str, Any]],
    *,
    planner: Callable[..., Any] | None = None,
    entailment: Callable[..., Any] | None = None,
    support_threshold: float = 0.70,
) -> dict[str, Any]:
    """Functional convenience API."""
    return EvidencePlan(planner, entailment, support_threshold=support_threshold).plan(question, contexts)


__all__ = [
    "ANSWER", "REFUSE", "SINGLE_DOC", "REQUIRES_MULTI_DOC", "INSUFFICIENT",
    "EVIDENCE_PLAN_UNAVAILABLE", "CoverageCell", "EvidencePlanResult", "EvidencePlan", "route_evidence",
]
