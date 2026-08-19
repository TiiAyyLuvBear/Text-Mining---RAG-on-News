from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, object]]], *, rrf_k: int = 60, top_k: int = 10
) -> list[dict[str, object]]:
    """Fuse ranked lists and retain the highest-ranked payload per chunk id."""
    fused: dict[str, dict[str, object]] = {}
    for results in ranked_lists:
        for rank, row in enumerate(results, start=1):
            chunk_id = str(row["chunk_id"])
            contribution = 1.0 / (rrf_k + rank)
            if chunk_id not in fused:
                fused[chunk_id] = {**row, "score": 0.0, "retrievers": []}
            fused[chunk_id]["score"] = float(fused[chunk_id]["score"]) + contribution
            fused[chunk_id]["retrievers"].append(rank)
    return sorted(fused.values(), key=lambda row: (-float(row["score"]), str(row["chunk_id"])))[:top_k]
