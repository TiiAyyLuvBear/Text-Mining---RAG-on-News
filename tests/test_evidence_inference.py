from src.backend import config
from src.backend.pipeline import NewsPipeline
import pytest


def chunk(article_id, score, text, chunk_id="1"):
    return {"article_id": article_id, "chunk_id": chunk_id, "text": text, "rerank_score": score}


def test_single_article_one_complete_chunk_can_pass_gate(monkeypatch):
    monkeypatch.setattr(config, "RERANK_MIN_SCORE", 1.0)
    result = NewsPipeline.evidence_quality_details([chunk("a", 2.0, "Đủ thông tin.")])
    assert result["status"] == "sufficient"
    assert result["article_count"] == 1
    assert result["legacy_margin"] == float("-inf")


def test_chunks_same_article_are_grouped_and_aggregated(monkeypatch):
    pipeline = NewsPipeline.__new__(NewsPipeline)
    ranked = [chunk("a", 3.0, "Phần một.", "1"), chunk("a", 2.5, "Phần hai.", "2")]
    monkeypatch.setattr(pipeline, "retrieve", lambda question: ranked)
    monkeypatch.setattr(pipeline, "rerank", lambda question, candidates: candidates)
    selected, sufficient, _, _ = pipeline.search_with_evidence("q", 2)
    assert sufficient is True
    assert len(selected) == 1
    assert selected[0]["article_chunk_count"] == 2
    assert "Phần một." in selected[0]["text"] and "Phần hai." in selected[0]["text"]


def test_two_supporting_articles_increase_corroboration_without_margin_penalty(monkeypatch):
    monkeypatch.setattr(config, "RERANK_MIN_SCORE", 1.0)
    result = NewsPipeline.evidence_quality_details([
        chunk("a", 3.0, "Cùng support."), chunk("b", 2.9, "Cùng support."),
    ])
    assert result["status"] == "sufficient"
    assert result["corroboration"] == 2
    assert result["legacy_margin"] == pytest.approx(0.1)


def test_partial_multi_article_pool_is_not_called_sufficient(monkeypatch):
    monkeypatch.setattr(config, "RERANK_MIN_SCORE", 1.0)
    monkeypatch.setattr(config, "RERANK_PARTIAL_MIN_SCORE", 0.0)
    result = NewsPipeline.evidence_quality_details([
        chunk("a", 0.8, "Một phần."), chunk("b", 0.7, "Phần khác."),
    ])
    assert result["status"] == "partial"
    assert result["article_count"] == 2


def test_explicit_opposite_polarity_is_conflicted(monkeypatch):
    monkeypatch.setattr(config, "RERANK_MIN_SCORE", 1.0)
    result = NewsPipeline.evidence_quality_details([
        chunk("a", 3.0, "Purin gây bệnh."), chunk("b", 2.8, "Purin không gây bệnh."),
    ])
    assert result["status"] == "conflicted"
    assert result["contradiction_detected"] is True


def test_empty_pool_is_insufficient():
    result = NewsPipeline.evidence_quality_details([])
    assert result["status"] == "insufficient"
    assert result["corroboration"] == 0
