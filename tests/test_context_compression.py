from src.backend.context_compression import (
    compress_context_by_sentence,
    pack_contexts_with_budget,
    score_sentence,
    sentence_split,
)


def word_count(text):
    return len(text.split())


def test_sentence_split_handles_empty_unicode_newline_and_bullets():
    assert sentence_split("") == []
    sentences = sentence_split(
        "  Việt Nam tăng trưởng.\n• Doanh thu tăng!\n- Lợi nhuận ổn định?  ",
        article_id="a",
        chunk_id="c",
        citation_rank=3,
    )
    assert [item["text"] for item in sentences] == [
        "Việt Nam tăng trưởng.",
        "Doanh thu tăng!",
        "Lợi nhuận ổn định?",
    ]
    assert [item["sentence_index"] for item in sentences] == [0, 1, 2]
    assert all(item["article_id"] == "a" and item["citation_rank"] == 3 for item in sentences)


def test_sentence_split_preserves_decimals_dates_abbreviations_urls_and_quotes():
    text = (
        'PGS.TS. An nói: "TP.HCM tăng 3.14% ngày 02.09.2025." '
        'Xem https://example.com/a.b để biết.'
    )
    assert [item["text"] for item in sentence_split(text)] == [
        'PGS.TS. An nói: "TP.HCM tăng 3.14% ngày 02.09.2025."',
        "Xem https://example.com/a.b để biết.",
    ]


def test_score_sentence_is_deterministic_and_uses_plan_signals():
    plan = {"entities": ["Công ty A"], "dates": ["2025"]}
    first = score_sentence("doanh thu năm 2025", "Công ty A có doanh thu năm 2025.", plan)
    second = score_sentence("doanh thu năm 2025", "Công ty A có doanh thu năm 2025.", plan)
    assert first == second
    assert first > score_sentence("doanh thu năm 2025", "Thời tiết hôm nay tốt.", plan)


def test_compression_prunes_by_threshold_falls_back_and_preserves_order_and_ids():
    contexts = [{
        "article_id": "a",
        "chunk_id": "c1",
        "citation_rank": 7,
        "text": "Thời tiết hôm nay tốt. Doanh thu năm 2025 tăng mạnh. Một câu không liên quan.",
    }]
    compressed, stats = compress_context_by_sentence(
        "doanh thu năm 2025",
        contexts,
        threshold=0.5,
        token_counter=word_count,
    )
    assert compressed[0]["text"] == "Doanh thu năm 2025 tăng mạnh."
    assert compressed[0]["article_id"] == "a"
    assert compressed[0]["chunk_id"] == "c1"
    assert compressed[0]["citation_rank"] == 7
    assert stats["kept_sentence_count"] == 1
    assert stats["tokens_after"] < stats["tokens_before"]

    fallback, _ = compress_context_by_sentence(
        "không có từ trùng",
        contexts,
        threshold=1.0,
    )
    assert fallback[0]["kept_sentence_count"] == 1


def test_empty_question_and_context_are_safe():
    compressed, stats = compress_context_by_sentence("", [], threshold=0.25)
    assert compressed == []
    assert stats["tokens_before"] == 0
    assert stats["compression_ratio"] == 0.0


def test_pack_never_exceeds_budget_and_handles_oversized_sentence():
    contexts = [{
        "article_id": "a",
        "chunk_id": "c1",
        "citation_rank": 1,
        "text": "một hai ba bốn năm sáu bảy tám",
    }]
    packed, stats = pack_contexts_with_budget(
        contexts,
        token_budget=4,
        token_counter=word_count,
        route_decision={"selected_article_ids": ["a"], "covered_sub_questions": []},
    )
    assert stats["tokens_packed"] <= 4
    assert packed == []
    assert stats["required_articles_missing"] == ["a"]


def test_multidoc_pack_preserves_required_sources_and_subquestion_coverage():
    contexts = [
        {
            "article_id": "a",
            "chunk_id": "ca",
            "citation_rank": 1,
            "text": "Doanh thu A 100 tỷ. Tin phụ A rất dài.",
        },
        {
            "article_id": "b",
            "chunk_id": "cb",
            "citation_rank": 2,
            "text": "Doanh thu B 80 tỷ. Tin phụ B rất dài.",
        },
    ]
    coverage = [
        {"sub_question_id": "sq1", "candidates": [{"article_id": "a", "chunk_id": "ca", "supports": True}]},
        {"sub_question_id": "sq2", "candidates": [{"article_id": "b", "chunk_id": "cb", "supports": True}]},
    ]
    compressed, _ = compress_context_by_sentence(
        "so sánh doanh thu A và B",
        contexts,
        coverage_matrix=coverage,
        threshold=0.25,
        token_counter=word_count,
    )
    packed, stats = pack_contexts_with_budget(
        compressed,
        token_budget=20,
        token_counter=word_count,
        route_decision={
            "route": "REQUIRES_MULTI_DOC",
            "selected_article_ids": ["a", "b"],
            "covered_sub_questions": ["sq1", "sq2"],
        },
    )
    assert [item["article_id"] for item in packed] == ["a", "b"]
    assert stats["tokens_packed"] <= 20
    assert stats["required_sub_questions_preserved"] == ["sq1", "sq2"]


def test_multidoc_pack_reserves_budget_when_first_source_has_oversized_sentence():
    contexts = [
        {"article_id": "a", "chunk_id": "ca", "citation_rank": 1, "text": "một hai ba bốn năm sáu bảy tám chín mười"},
        {"article_id": "b", "chunk_id": "cb", "citation_rank": 2, "text": "nguồn B ngắn"},
    ]
    packed, stats = pack_contexts_with_budget(
        contexts,
        token_budget=7,
        token_counter=word_count,
        route_decision={"selected_article_ids": ["a", "b"], "covered_sub_questions": []},
    )
    assert [item["article_id"] for item in packed] == ["b"]
    assert stats["required_articles_missing"] == ["a"]
    assert stats["tokens_packed"] <= 7


def test_coverage_protects_best_subquestion_sentence_from_threshold_pruning():
    contexts = [{
        "article_id": "a",
        "chunk_id": "ca",
        "citation_rank": 1,
        "text": "Doanh thu A tăng. Doanh thu B năm 2025 đạt 80 tỷ đồng.",
    }]
    plan = {
        "sub_questions": [
            {"id": "sq1", "text": "Doanh thu A tăng thế nào?"},
            {"id": "sq2", "text": "Doanh thu B năm 2025 là bao nhiêu?"},
        ]
    }
    coverage = [{
        "sub_question_id": "sq2",
        "candidates": [{"article_id": "a", "chunk_id": "ca", "supports": True}],
    }]
    compressed, _ = compress_context_by_sentence(
        "Doanh thu A tăng thế nào?",
        contexts,
        evidence_plan=plan,
        coverage_matrix=coverage,
        threshold=0.95,
    )
    assert "Doanh thu B năm 2025 đạt 80 tỷ đồng." in compressed[0]["text"]
    protected = [
        item for item in compressed[0]["_sentences"]
        if "sq2" in item["covered_sub_questions"]
    ]
    assert [item["text"] for item in protected] == ["Doanh thu B năm 2025 đạt 80 tỷ đồng."]


def test_three_source_pack_is_not_monopolized_by_high_score_article():
    contexts = [
        {
            "article_id": article_id,
            "chunk_id": f"c{index}",
            "citation_rank": index,
            "text": text,
            "_sentences": [{
                "article_id": article_id,
                "chunk_id": f"c{index}",
                "citation_rank": index,
                "sentence_index": 0,
                "context_index": index - 1,
                "text": text,
                "score": score,
                "covered_sub_questions": [f"sq{index}"],
            }],
        }
        for index, (article_id, text, score) in enumerate([
            ("a", "A cung cấp bằng chứng một.", 99.0),
            ("b", "B cung cấp bằng chứng hai.", 1.0),
            ("c", "C cung cấp bằng chứng ba.", 0.5),
        ], start=1)
    ]
    packed, stats = pack_contexts_with_budget(
        contexts,
        token_budget=30,
        token_counter=word_count,
        route_decision={
            "selected_article_ids": ["a", "b", "c"],
            "covered_sub_questions": ["sq1", "sq2", "sq3"],
        },
    )
    assert [item["article_id"] for item in packed] == ["a", "b", "c"]
    assert stats["required_sub_questions_missing"] == []
