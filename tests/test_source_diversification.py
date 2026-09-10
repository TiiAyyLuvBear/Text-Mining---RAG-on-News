import pytest

from src.backend.source_diversification import (
    diversification_benchmark,
    diversify_by_article,
    source_recall_at_k,
)


CANDIDATES = [
    {"chunk_id": "a1", "article_id": "A"},
    {"chunk_id": "a2", "article_id": "A"},
    {"chunk_id": "a3", "article_id": "A"},
    {"chunk_id": "b1", "article_id": "B"},
    {"chunk_id": "c1", "article_id": "C"},
]


@pytest.mark.parametrize(
    ("limit", "expected"),
    [
        (None, ["a1", "a2", "a3", "b1", "c1"]),
        (3, ["a1", "a2", "a3", "b1", "c1"]),
        (2, ["a1", "a2", "b1", "c1"]),
        (1, ["a1", "b1", "c1"]),
        (0, ["a1", "a2", "a3", "b1", "c1"]),
    ],
)
def test_diversification_caps_are_stable(limit, expected):
    assert [item["chunk_id"] for item in diversify_by_article(CANDIDATES, limit)] == expected


def test_missing_article_ids_are_unique_and_duplicate_chunks_are_safe():
    candidates = [
        {"chunk_id": "x", "text": "first"},
        {"chunk_id": "x", "text": "duplicate"},
        {"chunk_id": "y", "text": "anonymous second"},
    ]
    selected = diversify_by_article(candidates, 1)
    assert [item["chunk_id"] for item in selected] == ["x", "y"]


def test_metadata_is_preserved():
    candidate = {
        "chunk_id": "x",
        "article_id": "A",
        "text": "evidence",
        "rerank_score": 0.7,
        "rank": 4,
        "citation_rank": 7,
    }
    assert diversify_by_article([candidate], 1)[0] == candidate


def test_source_recall_empty_partial_and_full():
    assert source_recall_at_k(["A"], []) == 1.0
    assert source_recall_at_k(["X"], ["A"], 10) == 0.0
    assert source_recall_at_k(["A", "B"], ["A", "B"], 2) == 1.0
    assert source_recall_at_k(["A", "X"], ["A", "B"], 2) == 0.5
    report = diversification_benchmark(CANDIDATES)
    assert report["1"]["unique_articles"] == 3
