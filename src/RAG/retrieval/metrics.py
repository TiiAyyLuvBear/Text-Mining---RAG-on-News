from __future__ import annotations

import math


def metrics_for_ranking(relevant_chunk_ids: set[str], ranked_chunk_ids: list[str], ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in ks:
        retrieved = set(ranked_chunk_ids[:k])
        hits = len(retrieved & relevant_chunk_ids)
        metrics[f"recall@{k}"] = hits / len(relevant_chunk_ids) if relevant_chunk_ids else 0.0
        metrics[f"hit@{k}"] = 1.0 if hits else 0.0
    first_rank = next((rank for rank, chunk_id in enumerate(ranked_chunk_ids[:10], start=1) if chunk_id in relevant_chunk_ids), None)
    metrics["mrr@10"] = 1.0 / first_rank if first_rank else 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank, chunk_id in enumerate(ranked_chunk_ids[:10], start=1) if chunk_id in relevant_chunk_ids)
    ideal_count = min(len(relevant_chunk_ids), 10)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    metrics["ndcg@10"] = dcg / idcg if idcg else 0.0
    return metrics


def mean_metrics(per_query: list[dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {key: round(sum(row[key] for row in per_query) / len(per_query), 6) for key in keys}
