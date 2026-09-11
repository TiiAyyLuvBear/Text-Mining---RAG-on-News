"""Benchmark answer/refusal routing against QA output and review sidecars.

Supports raw ``QA_output`` records plus reviewed JSONL/CSV sidecars. Metrics
focus on safe deflection, not lexical answer similarity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


ANSWER = "ANSWER"
REFUSE = "REFUSE"
_TRUE = {"1", "true", "yes", "y", "answer", "answerable", "complete"}
_FALSE = {"0", "false", "no", "n", "refuse", "unanswerable", "missing", "noisy", "insufficient"}


def _records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def load_qa(output_path: str | Path, reviewed_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load QA output and merge review fields by id, qa_id, or question."""
    rows = _records(output_path)
    if not reviewed_path:
        return rows
    reviewed = _records(reviewed_path)
    index: dict[str, dict[str, Any]] = {}
    for item in reviewed:
        for key in ("id", "qa_id", "question"):
            value = str(item.get(key) or "").strip()
            if value:
                index[f"{key}:{value}"] = item
                break
    merged = []
    for row in rows:
        review = None
        for key in ("id", "qa_id", "question"):
            value = str(row.get(key) or "").strip()
            if value and f"{key}:{value}" in index:
                review = index[f"{key}:{value}"]
                break
        merged.append({**row, **(review or {})})
    return merged


def merge_predictions(rows: list[dict[str, Any]], predictions_path: str | Path | None) -> list[dict[str, Any]]:
    """Merge router outputs from JSONL/CSV prediction sidecar."""
    if not predictions_path:
        return rows
    predictions = _records(predictions_path)
    index: dict[str, dict[str, Any]] = {}
    for item in predictions:
        key = str(item.get("id") or item.get("qa_id") or item.get("question") or "").strip()
        if key:
            index[key] = item
    merged = []
    for row in rows:
        key = str(row.get("id") or row.get("qa_id") or row.get("question") or "").strip()
        merged.append({**row, **index.get(key, {})})
    return merged


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().casefold()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


def expected_answerable(row: Mapping[str, Any]) -> bool:
    """Resolve review label first, then conventional QA ``is_possible``."""
    for key in ("answerable", "is_answerable", "expected_answerable", "is_possible"):
        value = _bool(row.get(key))
        if value is not None:
            return value
    label = str(row.get("evidence_label") or row.get("coverage_label") or row.get("case") or "").casefold()
    if label:
        if label == "noisy":
            return True
        return label not in {"missing", "noisy", "insufficient", "unanswerable", "refuse"}
    return True


def expected_route(row: Mapping[str, Any]) -> str:
    label = str(row.get("evidence_label") or row.get("coverage_label") or row.get("case") or "").casefold()
    # Noisy fixture still contains complete evidence; gold answerability comes
    # from review/is_possible, while Missing means evidence absent.
    if label in {"missing", "insufficient", "unanswerable", "refuse"}:
        return REFUSE
    route = str(row.get("expected_decision") or row.get("gold_decision") or "").upper()
    if route:
        return REFUSE if route in {REFUSE, "INSUFFICIENT"} else ANSWER
    return ANSWER if expected_answerable(row) else REFUSE


def _prediction(row: Mapping[str, Any]) -> str:
    value: Any = row.get("decision") or row.get("route")
    generation = row.get("generation_decision")
    route_decision = row.get("route_decision")
    if isinstance(route_decision, Mapping):
        value = route_decision.get("route") or value
    if isinstance(generation, Mapping):
        value = generation.get("decision") or generation.get("route") or value
    elif isinstance(row.get("prediction"), Mapping):
        value = row["prediction"].get("decision") or row["prediction"].get("route") or value
    text = str(value or "").upper()
    if text in {REFUSE, "INSUFFICIENT", "REQUIRES_REFUSAL"}:
        return REFUSE
    return ANSWER


def _predicted_sources(row: Mapping[str, Any]) -> set[str]:
    value: Any = row.get("required_sources")
    route_decision = row.get("route_decision")
    if isinstance(route_decision, Mapping):
        value = route_decision.get("selected_article_ids", value)
    generation = row.get("generation_decision")
    if isinstance(generation, Mapping):
        value = generation.get("required_sources", value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            value = parsed if isinstance(parsed, list) else [value]
        except (TypeError, ValueError, json.JSONDecodeError):
            value = [value]
    return {str(item).strip() for item in (value or []) if str(item).strip()}


def _gold_sources(row: Mapping[str, Any]) -> set[str]:
    value: Any = row.get("source_article_ids") or row.get("required_sources") or row.get("article_id")
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            value = parsed if isinstance(parsed, list) else [value]
        except (TypeError, ValueError, json.JSONDecodeError):
            value = [part.strip() for part in value.split(",") if part.strip()]
    return {str(item).strip() for item in (value or []) if str(item).strip()}


def evaluate_predictions(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute deflection metrics and per-query error report."""
    rows = list(rows)
    details: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        expected = expected_route(row)
        actual = _prediction(row)
        # Shortcut: noisy retrieval answered using distractor source.
        label = str(row.get("evidence_label") or row.get("coverage_label") or row.get("case") or "").casefold()
        shortcut = (label == "noisy" and actual == ANSWER and bool(_gold_sources(row))
                    and not (_predicted_sources(row) & _gold_sources(row)))
        answerable = expected == ANSWER
        correct = expected == actual and not shortcut
        error_type = None
        if shortcut:
            error_type = "shortcut"
        elif not correct:
            if answerable and actual == REFUSE:
                error_type = "false_refusal"
            else:
                error_type = "hallucination_wrong_guess"
        details.append({
            "index": index,
            "id": row.get("id") or row.get("qa_id"),
            "question": row.get("question"),
            "expected": expected,
            "actual": actual,
            "correct": correct,
            "error_type": error_type,
            "shortcut": shortcut,
        })
    total_answerable = sum(item["expected"] == ANSWER for item in details)
    total_unanswerable = len(details) - total_answerable
    answered_correct = sum(item["expected"] == ANSWER and item["actual"] == ANSWER and not item["shortcut"] for item in details)
    refused_correct = sum(item["expected"] == REFUSE and item["actual"] == REFUSE for item in details)
    false_refusal_count = sum(item["error_type"] == "false_refusal" for item in details)
    wrong_guess_count = sum(item["expected"] == REFUSE and item["actual"] == ANSWER for item in details)
    shortcut_count = sum(item["shortcut"] for item in details)
    answerable_accuracy = answered_correct / total_answerable if total_answerable else 0.0
    deflection_accuracy = refused_correct / total_unanswerable if total_unanswerable else 0.0
    adt = (2 * answerable_accuracy * deflection_accuracy / (answerable_accuracy + deflection_accuracy)
           if answerable_accuracy + deflection_accuracy else 0.0)
    by_case: dict[str, dict[str, Any]] = {}
    for item, row in zip(details, rows if isinstance(rows, list) else []):
        label = str(row.get("evidence_label") or row.get("coverage_label") or row.get("case") or "unknown").casefold()
        group = by_case.setdefault(label, {"count": 0, "correct": 0, "errors": 0})
        group["count"] += 1
        group["correct"] += int(item["correct"])
        group["errors"] += int(not item["correct"])
    for group in by_case.values():
        group["accuracy"] = round(group["correct"] / group["count"], 6) if group["count"] else 0.0
    return {
        "count": len(details),
        "answerable_count": total_answerable,
        "unanswerable_count": total_unanswerable,
        "answerable_accuracy": round(answerable_accuracy, 6),
        "deflection_accuracy": round(deflection_accuracy, 6),
        "ADT": round(adt, 6),
        "adt_harmonic": round(adt, 6),
        "Answerable Accuracy": round(answerable_accuracy, 6),
        "Deflection Accuracy": round(deflection_accuracy, 6),
        "ADT harmonic": round(adt, 6),
        "false_refusal": round(false_refusal_count / total_answerable, 6) if total_answerable else 0.0,
        "false_refusal_rate": round(false_refusal_count / total_answerable, 6) if total_answerable else 0.0,
        "false_refusal_count": false_refusal_count,
        "shortcut": round(shortcut_count / len(details), 6) if details else 0.0,
        "shortcut_rate": round(shortcut_count / len(details), 6) if details else 0.0,
        "shortcut_count": shortcut_count,
        "hallucination_wrong_guess": round(wrong_guess_count / total_unanswerable, 6) if total_unanswerable else 0.0,
        "hallucination_rate": round(wrong_guess_count / total_unanswerable, 6) if total_unanswerable else 0.0,
        "hallucination_wrong_guess_count": wrong_guess_count,
        "per_query_errors": [item for item in details if item["error_type"]],
        "per_query": details,
        "by_case": by_case,
        "group_metrics": by_case,
    }


def compute_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Alias for callers using metric terminology."""
    return evaluate_predictions(rows)


def run_benchmark(
    output_path: str | Path,
    reviewed_path: str | Path | None = None,
    predictions_path: str | Path | None = None,
) -> dict[str, Any]:
    return evaluate_predictions(merge_predictions(load_qa(output_path, reviewed_path), predictions_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="QA_output JSONL/CSV")
    parser.add_argument("--reviewed", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--errors", type=Path, help="Write per-query errors JSONL")
    args = parser.parse_args()
    result = run_benchmark(args.output, args.reviewed, args.predictions)
    print(json.dumps({key: value for key, value in result.items() if key not in {"per_query", "per_query_errors"}}, ensure_ascii=False, indent=2))
    if args.errors:
        args.errors.parent.mkdir(parents=True, exist_ok=True)
        with args.errors.open("w", encoding="utf-8") as handle:
            for row in result["per_query_errors"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
