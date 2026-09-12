"""End-to-end QA benchmark for answers, retrieval, and evidence routing.

Runs the public QA endpoint or evaluates saved endpoint responses. Answer quality
uses deterministic gold-answer similarity; it is a regression metric, not an LLM
factuality judge. Retrieval metrics use QA gold article/chunk IDs. Routing treats
``is_possible=false`` as a required refusal.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

ANSWER_STATUSES = {"generated", "extractive_fallback"}
_TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [part.strip() for part in value.split(",")]
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def gold_article_ids(row: Mapping[str, Any]) -> list[str]:
    return _ids(row.get("source_article_ids")) or _ids(row.get("article_id"))


def _lcs_length(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_word in left:
        current = [0]
        for index, right_word in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if left_word == right_word else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _answer_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", str(text or "")).lower()
    return [token for token in _TOKEN_RE.findall(normalized) if len(token) > 1]


def answer_similarity(answer: str, gold_answers: Iterable[str]) -> dict[str, float]:
    """Return best token F1, gold coverage, and ROUGE-L across gold variants."""
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]
    answer_tokens = _answer_tokens(answer)
    answer_counts = Counter(answer_tokens)
    best = {"token_f1": 0.0, "gold_token_recall": 0.0, "rouge_l": 0.0, "exact_match": 0.0}
    for gold in gold_answers:
        gold_tokens = _answer_tokens(gold)
        if not gold_tokens:
            continue
        overlap = sum((answer_counts & Counter(gold_tokens)).values())
        precision = overlap / len(answer_tokens) if answer_tokens else 0.0
        recall = overlap / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        lcs = _lcs_length(answer_tokens, gold_tokens)
        rouge_precision = lcs / len(answer_tokens) if answer_tokens else 0.0
        rouge_recall = lcs / len(gold_tokens)
        rouge_l = (2 * rouge_precision * rouge_recall / (rouge_precision + rouge_recall)
                   if rouge_precision + rouge_recall else 0.0)
        candidate = {
            "token_f1": f1,
            "gold_token_recall": recall,
            "rouge_l": rouge_l,
            "exact_match": float(answer_tokens == gold_tokens),
        }
        if candidate["token_f1"] > best["token_f1"]:
            best = candidate
    return {key: round(value, 6) for key, value in best.items()}


def _contexts(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("contexts", "retrieval"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def _predicted_article_ids(payload: Mapping[str, Any]) -> list[str]:
    route = payload.get("route_decision")
    if isinstance(route, Mapping):
        selected = _ids(route.get("selected_article_ids"))
        if selected:
            return selected
    return list(dict.fromkeys(
        str(item.get("article_id")).strip() for item in _contexts(payload) if item.get("article_id") is not None
    ))


def _predicted_chunk_ids(payload: Mapping[str, Any]) -> list[str]:
    return list(dict.fromkeys(
        str(item.get("chunk_id")).strip() for item in _contexts(payload) if item.get("chunk_id") is not None
    ))


def _decision(payload: Mapping[str, Any]) -> str:
    decision = payload.get("generation_decision")
    if isinstance(decision, Mapping) and str(decision.get("decision") or "").upper() in {"ANSWER", "REFUSE"}:
        return str(decision["decision"]).upper()
    status = str(payload.get("answer_status") or "").casefold()
    return "ANSWER" if status in ANSWER_STATUSES else "REFUSE"


def _route(payload: Mapping[str, Any]) -> str:
    decision = payload.get("route_decision")
    if isinstance(decision, Mapping):
        route = str(decision.get("route") or "").upper()
        if route in {"SINGLE_DOC", "REQUIRES_MULTI_DOC", "INSUFFICIENT"}:
            return route
    return "INSUFFICIENT" if _decision(payload) == "REFUSE" else "SINGLE_DOC"


def _rank_metrics(gold: set[str], ranked: list[str]) -> dict[str, float]:
    first_rank = next((index for index, item in enumerate(ranked, start=1) if item in gold), None)
    return {
        "hit_at_1": float(bool(set(ranked[:1]) & gold)),
        "hit_at_5": float(bool(set(ranked[:5]) & gold)),
        "recall_at_5": len(set(ranked[:5]) & gold) / len(gold) if gold else 0.0,
        "mrr_at_5": 1 / first_rank if first_rank and first_rank <= 5 else 0.0,
    }


def evaluate_payload(qa: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    answerable = bool(qa.get("is_possible", True))
    gold_articles = gold_article_ids(qa)
    gold_chunks = set(_ids(qa.get("gold_id") or qa.get("gold_chunk_ids")))
    expected_decision = "ANSWER" if answerable else "REFUSE"
    expected_route = ("INSUFFICIENT" if not answerable else
                      "REQUIRES_MULTI_DOC" if len(gold_articles) > 1 else "SINGLE_DOC")
    actual_decision = _decision(payload)
    actual_route = _route(payload)
    selected_articles = _predicted_article_ids(payload)
    retrieval_articles = [str(item.get("article_id")) for item in _contexts(payload) if item.get("article_id") is not None]
    answer = str(payload.get("answer") or "")
    status = str(payload.get("answer_status") or "")
    quality = answer_similarity(answer, qa.get("answers") or []) if answerable and actual_decision == "ANSWER" else None
    return {
        "qa_id": str(qa.get("id") or qa.get("qa_id")),
        "question": str(qa.get("question") or ""),
        "qa_type": qa.get("qa_type"),
        "answerable": answerable,
        "gold_article_ids": gold_articles,
        "gold_chunk_ids": sorted(gold_chunks),
        "expected_decision": expected_decision,
        "actual_decision": actual_decision,
        "decision_correct": actual_decision == expected_decision,
        "expected_route_heuristic": expected_route,
        "actual_route": actual_route,
        "route_correct": actual_route == expected_route,
        "answer_status": status,
        "answer": answer,
        "answer_quality": quality,
        "retrieval_article": _rank_metrics(set(gold_articles), retrieval_articles) if answerable else None,
        "retrieval_chunk": _rank_metrics(gold_chunks, _predicted_chunk_ids(payload)) if answerable and gold_chunks else None,
        "selected_source_recall": (len(set(selected_articles) & set(gold_articles)) / len(gold_articles)
                                   if answerable and gold_articles else None),
        "selected_article_ids": selected_articles,
        "response_time_ms": payload.get("response_time_ms"),
        "error": payload.get("benchmark_error"),
    }


def _mean(rows: Iterable[Mapping[str, Any]], path: tuple[str, ...]) -> float:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value.get(key) if isinstance(value, Mapping) else None
        if isinstance(value, (int, float)):
            values.append(float(value))
    return round(statistics.mean(values), 6) if values else 0.0


def summarize(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in records if row["answerable"]]
    unanswerable = [row for row in records if not row["answerable"]]
    response_times = [float(row["response_time_ms"]) for row in records if isinstance(row.get("response_time_ms"), (int, float))]
    quality_rows = [row for row in answerable if row.get("answer_quality")]
    return {
        "count": len(records),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "routing": {
            "decision_accuracy": _mean(records, ("decision_correct",)),
            "answerable_answer_rate": _mean(answerable, ("decision_correct",)),
            "unanswerable_deflection_rate": _mean(unanswerable, ("decision_correct",)),
            "route_class_accuracy_heuristic": _mean(records, ("route_correct",)),
            "false_refusal_count": sum(row["answerable"] and row["actual_decision"] == "REFUSE" for row in records),
            "wrong_answer_count": sum(not row["answerable"] and row["actual_decision"] == "ANSWER" for row in records),
        },
        "retrieval_answerable_only": {
            "article_hit_at_1": _mean(answerable, ("retrieval_article", "hit_at_1")),
            "article_hit_at_5": _mean(answerable, ("retrieval_article", "hit_at_5")),
            "article_recall_at_5": _mean(answerable, ("retrieval_article", "recall_at_5")),
            "article_mrr_at_5": _mean(answerable, ("retrieval_article", "mrr_at_5")),
            "chunk_hit_at_5": _mean(answerable, ("retrieval_chunk", "hit_at_5")),
            "selected_source_recall": _mean(answerable, ("selected_source_recall",)),
        },
        "answer_quality_answered_only": {
            "evaluated_count": len(quality_rows),
            "token_f1": _mean(quality_rows, ("answer_quality", "token_f1")),
            "gold_token_recall": _mean(quality_rows, ("answer_quality", "gold_token_recall")),
            "rouge_l": _mean(quality_rows, ("answer_quality", "rouge_l")),
            "exact_match": _mean(quality_rows, ("answer_quality", "exact_match")),
        },
        "latency": {
            "mean_response_time_ms": round(statistics.mean(response_times), 3) if response_times else 0.0,
            "p95_response_time_ms": round(sorted(response_times)[max(0, int(len(response_times) * .95) - 1)], 3) if response_times else 0.0,
        },
        "errors": sum(bool(row.get("error")) for row in records),
    }


def benchmark(
    qa_rows: Iterable[Mapping[str, Any]],
    submit: Callable[[str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    for qa in qa_rows:
        qa_id = str(qa.get("id") or qa.get("qa_id"))
        try:
            payload = dict(submit(str(qa.get("question") or "")))
        except Exception as exc:  # Keep benchmark complete when one request fails.
            payload = {"answer_status": "benchmark_error", "benchmark_error": type(exc).__name__}
        responses.append({"qa_id": qa_id, "payload": payload})
        records.append(evaluate_payload(qa, payload))
    return records, responses


def _http_submitter(api_url: str, top_k: int, timeout: float) -> Callable[[str], Mapping[str, Any]]:
    import requests

    endpoint = api_url.rstrip("/") + "/api/qa/ask"

    def submit(question: str) -> Mapping[str, Any]:
        response = requests.post(endpoint, json={"question": question, "top_k": top_k}, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("QA endpoint returned non-object JSON")
        return payload

    return submit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, default=Path("Dataset/qa_dataset.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/qa_benchmark"))
    parser.add_argument("--responses", type=Path, help="Reuse responses.jsonl from a prior endpoint run")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    qa_rows = read_jsonl(args.qa)
    if args.limit is not None:
        qa_rows = qa_rows[:max(0, args.limit)]
    if args.responses:
        payloads = {str(row["qa_id"]): row.get("payload", {}) for row in read_jsonl(args.responses)}
        records = [evaluate_payload(qa, payloads.get(str(qa.get("id")), {"benchmark_error": "missing_response"})) for qa in qa_rows]
        responses = [{"qa_id": str(qa.get("id")), "payload": payloads.get(str(qa.get("id")), {})} for qa in qa_rows]
    else:
        records, responses = benchmark(qa_rows, _http_submitter(args.api_url, args.top_k, args.timeout))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "responses.jsonl", responses)
    write_jsonl(args.output_dir / "per_query.jsonl", records)
    write_jsonl(args.output_dir / "failures.jsonl", [row for row in records if not row["decision_correct"] or row.get("error")])
    summary = {
        "qa_path": str(args.qa), "responses_path": str(args.responses) if args.responses else None,
        "api_url": None if args.responses else args.api_url, "top_k": args.top_k,
        "metrics": summarize(records),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
