from types import SimpleNamespace

import pytest

from src.backend import config
from src.backend.pipeline import NewsPipeline


def _result(chunk_id: str, **extra):
    return {"chunk_id": chunk_id, "article_id": chunk_id, "text": chunk_id, **extra}


def test_rrf_promotes_chunk_returned_by_dense_and_bm25():
    results = NewsPipeline._reciprocal_rank_fusion(
        [_result("dense-only"), _result("shared")],
        [_result("bm25-only"), _result("shared")],
        limit=3,
        rrf_k=60,
    )

    assert results[0]["chunk_id"] == "shared"
    assert results[0]["retrievers"] == ["dense", "bm25"]
    assert results[0]["retrieval_rank"] == 1
    assert results[0]["retrieval_score"] == pytest.approx(2 / 62)


def test_retrieve_runs_both_retrievers_and_returns_fused_candidates(monkeypatch):
    pipeline = NewsPipeline.__new__(NewsPipeline)
    dense_hits = [
        SimpleNamespace(payload=_result("dense-only"), score=0.9),
        SimpleNamespace(payload=_result("shared"), score=0.8),
    ]

    class Client:
        def query_points(self, **kwargs):
            assert kwargs["limit"] == 7
            return SimpleNamespace(points=dense_hits)

    class Vector:
        def tolist(self):
            return [0.1, 0.2]

    class Encoder:
        def encode(self, *args, **kwargs):
            return [Vector()]

    class BM25:
        def get_scores(self, tokens):
            assert tokens == ["shared"]
            return [3.0, 2.0, 0.0]

    rows = [_result("bm25-only"), _result("shared"), _result("zero-score")]
    pipeline.client = Client()
    pipeline._load_encoder = lambda: Encoder()
    pipeline._load_bm25_index = lambda: {"index": BM25(), "rows": rows}
    pipeline.is_ready = lambda: True
    monkeypatch.setattr(config, "HYBRID_CANDIDATE_K", 7)
    monkeypatch.setattr(config, "HYBRID_RRF_K", 60)
    monkeypatch.setattr(
        "src.RAG.retrieval.tokenize.tokenize_vietnamese", lambda question: [question]
    )

    results = pipeline.retrieve("shared", limit=3)

    assert results[0]["chunk_id"] == "shared"
    assert results[0]["dense_score"] == 0.8
    assert results[0]["bm25_score"] == 2.0
    assert "zero-score" not in {item["chunk_id"] for item in results}
