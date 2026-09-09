"""Source-level diversification and recall metrics for retrieval results."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def _article_id(candidate: dict[str, Any], index: int) -> str:
    value = str(candidate.get("article_id") or "").strip()
    return value or f"__anonymous_{index}"


def diversify_by_article(
    candidates: Iterable[dict[str, Any]],
    max_per_article: int | None = 1,
) -> list[dict[str, Any]]:
    """Keep ranked candidates while limiting chunks contributed by an article.

    ``None`` or a non-positive value means unlimited.  Input order is treated
    as rank order and is preserved; no score assumptions are imposed here.
    """

    items = list(candidates)
    if max_per_article is None or max_per_article <= 0:
        return items
    counts: defaultdict[str, int] = defaultdict(int)
    selected: list[dict[str, Any]] = []
    for index, candidate in enumerate(items):
        article_id = _article_id(candidate, index)
        if counts[article_id] >= max_per_article:
            continue
        counts[article_id] += 1
        selected.append(candidate)
    return selected


def source_recall_at_k(
    retrieved_sources: Iterable[Any],
    required_sources: Iterable[Any],
    k: int | None = None,
) -> float:
    """Return the fraction of required article/source IDs found in top-k.

    Duplicate retrieved IDs do not inflate recall.  An empty requirement is
    defined as 1.0 because there is nothing to miss.
    """

    required = {str(value).strip() for value in required_sources if str(value).strip()}
    if not required:
        return 1.0
    retrieved = list(retrieved_sources)
    if k is not None:
        retrieved = retrieved[: max(0, k)]
    found = {str(value).strip() for value in retrieved if str(value).strip()}
    return len(required & found) / len(required)

