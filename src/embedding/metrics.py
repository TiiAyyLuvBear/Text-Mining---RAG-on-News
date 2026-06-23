"""Ranking metrics for retrieval evaluation (article-level relevance)."""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Set


def _first_occurrence_unique(items: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def recall_at_k(ranked_articles: Sequence[str], gold: Set[str], k: int) -> float:
    if not gold:
        return 0.0
    topk = set(ranked_articles[:k])
    return len(topk & gold) / len(gold)


def hit_at_k(ranked_articles: Sequence[str], gold: Set[str], k: int) -> float:
    if not gold:
        return 0.0
    return 1.0 if set(ranked_articles[:k]) & gold else 0.0


def mrr_at_k(ranked_articles: Sequence[str], gold: Set[str], k: int) -> float:
    if not gold:
        return 0.0
    for idx, art in enumerate(ranked_articles[:k], start=1):
        if art in gold:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(ranked_articles: Sequence[str], gold: Set[str], k: int) -> float:
    if not gold:
        return 0.0
    dcg = 0.0
    for idx, art in enumerate(ranked_articles[:k], start=1):
        if art in gold:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_query(
    ranked_articles: Sequence[str],
    gold: Set[str],
    ks: Iterable[int] = (1, 5, 10),
) -> dict:
    unique = _first_occurrence_unique(ranked_articles)
    return {
        "ndcg@10": ndcg_at_k(unique, gold, 10),
        "recall@5": recall_at_k(unique, gold, 5),
        "recall@10": recall_at_k(unique, gold, 10),
        "mrr@10": mrr_at_k(unique, gold, 10),
        "hit@1": hit_at_k(unique, gold, 1),
        "hit@5": hit_at_k(unique, gold, 5),
    }