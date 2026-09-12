from src.backend.query_planner import build_evidence_plan
from src.backend.temporal_retrieval import (
    apply_temporal_boost,
    extract_temporal_terms,
    temporal_terms_from_plan,
)


def test_temporal_parser_handles_vietnamese_formats_and_ranges():
    question = (
        "trước năm 2020, từ tháng 3 đến tháng 7, "
        "ngày 2 tháng 9, 05/2025 và 02.09.2025"
    )
    assert extract_temporal_terms(question) == [
        "trước năm 2020",
        "từ tháng 3 đến tháng 7",
        "ngày 2 tháng 9",
        "05/2025",
        "02.09.2025",
    ]


def test_year_month_and_no_temporal_expression():
    assert extract_temporal_terms("Tin tháng 5 năm 2025") == ["tháng 5 năm 2025"]
    assert extract_temporal_terms("trước khi sự kiện X bắt đầu") == [
        "trước khi sự kiện x bắt đầu"
    ]
    assert extract_temporal_terms("Doanh thu 10 tỷ đồng") == []


def test_plan_is_the_preferred_temporal_contract_with_raw_fallback():
    plan = build_evidence_plan("Diễn biến từ tháng 5 đến tháng 7")
    assert temporal_terms_from_plan(plan) == ["từ tháng 5 đến tháng 7"]
    assert temporal_terms_from_plan({}, "Tin tức năm 2025") == ["năm 2025"]


def test_temporal_boost_is_soft_stable_and_preserves_nonmatches():
    candidates = [
        {"chunk_id": "a", "article_id": "a", "text": "không có ngày", "rerank_score": 0.90},
        {"chunk_id": "b", "article_id": "b", "text": "báo cáo năm 2025", "rerank_score": 0.87},
        {"chunk_id": "c", "article_id": "c", "text": "tin khác", "rerank_score": 0.80},
    ]
    ranked = apply_temporal_boost(candidates, ["năm 2025"], boost=0.05)
    assert [item["chunk_id"] for item in ranked] == ["b", "a", "c"]
    assert len(ranked) == 3
    assert ranked[0]["temporal_terms_matched"] == ["năm 2025"]
    assert ranked[1]["temporal_boost"] == 0.0
