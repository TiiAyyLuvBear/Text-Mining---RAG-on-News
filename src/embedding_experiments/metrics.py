from __future__ import annotations

import math
from collections.abc import Sequence


def retrieval_metrics(
    ranked_article_ids: Sequence[str],
    relevant_article_ids: set[str],
    *,
    max_k: int = 10,
) -> dict[str, float]:
    ranked = list(ranked_article_ids)[:max_k]
    gains = [1.0 if article_id in relevant_article_ids else 0.0 for article_id in ranked]
    return {
        "ndcg@10": ndcg_at_k(gains, min(10, max_k), relevant_count=len(relevant_article_ids)),
        "recall@5": recall_at_k(ranked, relevant_article_ids, 5),
        "recall@10": recall_at_k(ranked, relevant_article_ids, 10),
        "mrr@10": mrr_at_k(ranked, relevant_article_ids, 10),
        "hit@1": hit_at_k(ranked, relevant_article_ids, 1),
        "hit@5": hit_at_k(ranked, relevant_article_ids, 5),
    }


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def mrr_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    for index, article_id in enumerate(ranked[:k], start=1):
        if article_id in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(gains: Sequence[float], k: int, *, relevant_count: int) -> float:
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains[:k]))
    ideal_hits = min(relevant_count, k)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def average_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: round(sum(row[key] for row in rows) / len(rows), 6) for key in keys}

