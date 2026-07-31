from __future__ import annotations

import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

from .hybrid import reciprocal_rank_fusion
from .metrics import mean_metrics, metrics_for_ranking
from .schema import read_jsonl


Searcher = Callable[[str, int], list[dict[str, object]]]
METRIC_COLUMNS = ("recall@1", "recall@5", "recall@10", "hit@1", "hit@5", "hit@10", "mrr@10", "ndcg@10")


def evaluate_retrievers(
    qrels_path: str | Path,
    searchers: dict[str, Searcher],
    *,
    candidate_k: int = 50,
    ks: tuple[int, ...] = (1, 5, 10),
    show_progress: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    qrels = read_jsonl(qrels_path)
    per_query: list[dict[str, object]] = []
    aggregate: dict[str, list[dict[str, float]]] = {method: [] for method in searchers}
    latencies: dict[str, list[float]] = {method: [] for method in searchers}
    iterator = qrels
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(qrels, desc="Evaluating retrieval", unit="QA")
        except ImportError:  # pragma: no cover
            pass
    for qrel in iterator:
        relevant = set(map(str, qrel["relevant_chunk_ids"]))
        question = str(qrel["question"])
        raw_results: dict[str, list[dict[str, object]]] = {}
        for method, searcher in searchers.items():
            started = time.perf_counter()
            raw_results[method] = searcher(question, candidate_k)
            elapsed_ms = (time.perf_counter() - started) * 1000
            latencies[method].append(elapsed_ms)
            ranked = [str(row["chunk_id"]) for row in raw_results[method]]
            row_metrics = metrics_for_ranking(relevant, ranked, ks)
            aggregate[method].append(row_metrics)
            per_query.append({"qa_id": qrel["qa_id"], "method": method, "question": question, "article_id": qrel["article_id"], "relevant_chunk_ids": json.dumps(sorted(relevant), ensure_ascii=False), "retrieved_chunk_ids": json.dumps(ranked, ensure_ascii=False), "latency_ms": round(elapsed_ms, 4), **row_metrics})
    summary: dict[str, object] = {"qrels": len(qrels), "label_type": "weak_answer_chunk_match", "methods": {}}
    for method in searchers:
        values = latencies[method]
        summary["methods"][method] = {**mean_metrics(aggregate[method]), "mean_latency_ms": round(statistics.mean(values), 4) if values else 0.0, "p50_latency_ms": round(statistics.median(values), 4) if values else 0.0, "p95_latency_ms": round(sorted(values)[max(0, int(len(values) * .95) - 1)], 4) if values else 0.0}
    return summary, per_query


def write_evaluation(output_dir: str | Path, summary: dict[str, object], per_query: list[dict[str, object]], manifest: dict[str, object]) -> dict[str, str]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    summary_path, per_query_path, manifest_path, readme_path = (directory / "summary.json", directory / "per_query.csv", directory / "manifest.json", directory / "README.md")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with per_query_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_query[0]) if per_query else ["qa_id", "method"])
        writer.writeheader()
        writer.writerows(per_query)
    metric_columns = ["method", *METRIC_COLUMNS, "mean_latency_ms", "p50_latency_ms", "p95_latency_ms"]
    with (directory / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_columns)
        writer.writeheader()
        writer.writerows({"method": name, **values} for name, values in summary["methods"].items())
    readme_path.write_text("# Retrieval evaluation\n\nChunk relevance labels are weak labels inferred from QA answers; they are not manually annotated ground truth.\n", encoding="utf-8")
    return {"summary": str(summary_path), "metrics": str(directory / "metrics.csv"), "per_query": str(per_query_path), "manifest": str(manifest_path), "readme": str(readme_path)}


def read_per_query_csv(paths: list[str | Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (str(row["qa_id"]), str(row["method"]))
                if key in seen:
                    raise ValueError(f"Duplicate QA/method row while merging: {key}")
                seen.add(key)
                rows.append(dict(row))
    return rows


def summarize_per_query(per_query: list[dict[str, object]]) -> dict[str, object]:
    if not per_query:
        raise ValueError("Cannot summarize an empty per-query result set.")
    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in per_query:
        by_method[str(row["method"])].append(row)
    qa_ids = {method: {str(row["qa_id"]) for row in rows} for method, rows in by_method.items()}
    if len({frozenset(ids) for ids in qa_ids.values()}) != 1:
        raise ValueError("Every retrieval method must contain results for the same QA IDs.")
    summary: dict[str, object] = {"qrels": len(next(iter(qa_ids.values()))), "label_type": "weak_answer_chunk_match", "methods": {}}
    for method, rows in by_method.items():
        latencies = [float(row["latency_ms"]) for row in rows]
        metrics = {key: round(statistics.mean(float(row[key]) for row in rows), 6) for key in METRIC_COLUMNS}
        summary["methods"][method] = {
            **metrics,
            "mean_latency_ms": round(statistics.mean(latencies), 4),
            "p50_latency_ms": round(statistics.median(latencies), 4),
            "p95_latency_ms": round(sorted(latencies)[max(0, int(len(latencies) * .95) - 1)], 4),
        }
    return summary
