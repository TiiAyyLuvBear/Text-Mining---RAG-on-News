from src.RAG.retrieval.schema import EvidencePlan
from src.backend.query_planner import build_evidence_plan, normalize_question


def test_normalization_handles_unicode_whitespace_and_none():
    decomposed = "  Ho\u0300a   Pha\u0301t?  "
    assert normalize_question(decomposed) == "Hòa Phát?"
    assert normalize_question(None) == ""


def test_direct_factoid_is_stable_and_uses_shared_schema():
    plan = build_evidence_plan("Ai là CEO của X?")
    assert isinstance(plan, EvidencePlan)
    assert plan.answer_operator == "DIRECT"
    assert [item.id for item in plan.sub_questions] == ["sq1"]
    assert plan.entities == ["CEO", "X"]


def test_comparison_preserves_metric_entities_and_period():
    plan = build_evidence_plan("So sánh doanh thu A và B năm 2025")
    assert plan.answer_operator == "COMPARE"
    assert plan.entities == ["A", "B"]
    assert plan.dates == ["năm 2025"]
    assert plan.estimated_sources_needed == 2
    assert [item.id for item in plan.sub_questions] == ["sq1", "sq2"]
    assert [item.text for item in plan.sub_questions] == [
        "doanh thu A năm 2025",
        "doanh thu B năm 2025",
    ]


def test_timeline_causal_and_single_document_relation_cases():
    timeline = build_evidence_plan("Diễn biến sự kiện X từ tháng 5 đến tháng 7")
    causal = build_evidence_plan("Vì sao giá cổ phiếu giảm?")
    relation = build_evidence_plan("Vai trò của A và B trong sự kiện X là gì?")
    
    assert timeline.answer_operator == "TIMELINE"
    assert timeline.temporal_constraints == ["từ tháng 5 đến tháng 7"]
    assert causal.answer_operator == "CAUSAL_SUMMARY"
    assert relation.answer_operator == "DIRECT"
    assert relation.estimated_sources_needed == 1


def test_empty_input_fallback_never_crashes():
    plan = build_evidence_plan("")
    assert plan.normalized_question == ""
    assert plan.sub_questions[0].id == "sq1"


def test_sub_question_ids_are_repeatable_and_unique():
    first = build_evidence_plan("So sánh A và B")
    second = build_evidence_plan("So sánh A và B")
    first_ids = [item.id for item in first.sub_questions]
    
    assert first_ids == [item.id for item in second.sub_questions]
    assert len(first_ids) == len(set(first_ids))


def test_remote_branch_intent_improvements_are_preserved_without_forcing_route():
    claim = build_evidence_plan("Đúng hay sai: Trái Đất quay quanh Mặt Trời")
    timeline = build_evidence_plan("Trình tự sự kiện X qua các thời kỳ")
    explicit_sources = build_evidence_plan("Tóm tắt thông tin từ các nguồn")
    
    assert claim.sub_questions[0].text.startswith("Kiểm chứng thông tin:")
    assert timeline.answer_operator == "TIMELINE"
    assert explicit_sources.estimated_sources_needed == 3


def test_numbers_keep_units_and_do_not_leak_from_dates():
    plan = build_evidence_plan(
        "Số liệu 10 10.5 10,5 10% 10 tỷ vào 05/2025 và 02.09.2025"
    )
    assert plan.numbers == ["10", "10.5", "10,5", "10%", "10 tỷ"]
    assert plan.dates == ["05/2025", "02.09.2025"]


def test_real_temporal_comparison_splits_the_two_organisations_not_page_labels():
    plan = build_evidence_plan(
        "So sánh thời gian đăng ký xét tuyển trên Cổng thông tin tuyển sinh của "
        "Bộ GD-ĐT giữa Trường ĐH Công nghệ Giao thông vận tải năm 2024 và "
        "Trường ĐH Ngoại thương năm 2025. Có sự khác biệt nào về thời hạn đăng ký?"
    )

    assert len(plan.sub_questions) == 2
    assert "Trường ĐH Công nghệ Giao thông vận tải năm 2024" in plan.sub_questions[0].text
    assert "Trường ĐH Ngoại thương năm 2025" in plan.sub_questions[1].text
    assert all("(đối tượng: Cổng)" not in item.text for item in plan.sub_questions)


def test_subjective_cross_article_ranking_requires_direct_comparative_evidence():
    plan = build_evidence_plan(
        "So sánh quan điểm của ba bài báo: bài về tuổi gia chủ, bài về ngày giờ "
        "động thổ, và bài về kiêng kỵ thiết kế nhà. Yếu tố nào gây hậu quả "
        "nghiêm trọng nhất nếu vi phạm?"
    )

    assert len(plan.sub_questions) == 4
    assert plan.sub_questions[-1].evidence_type == "COMPARATIVE_CONCLUSION"


def test_passive_compared_to_which_model_is_a_single_relation_lookup():
    # Edge case from teammate: A question asking *what* a model is compared to is NOT a COMPARE operator.
    plan = build_evidence_plan(
        "Mức giá hiện tại của chiếc Mercedes-Benz C180 cũ đang được so sánh "
        "ngang bằng với mẫu xe tay ga nào trên thị trường?"
    )
    
    assert plan.answer_operator == "DIRECT"
    assert len(plan.sub_questions) == 1
    assert plan.sub_questions[0].required_entities


def test_explicit_date_range_is_timeline_even_without_timeline_keyword():
    # Edge case from teammate: Sentences structured with "Từ năm... đến năm..." are inherently timelines.
    plan = build_evidence_plan(
        "Từ năm 2023 đến năm 2025, kỳ thi tuyển sinh lớp 10 thay đổi thế nào?"
    )
    
    assert plan.answer_operator == "TIMELINE"
    assert plan.sub_questions[0].required_dates


def test_compression_preserves_one_mandatory_sentence_per_covered_subquestion():
    # Edge case from teammate: Ensures context compression logic handles sub-questions correctly.
    from src.backend.context_compression import (
        compress_context_by_sentence,
        pack_contexts_with_budget,
    )

    contexts = [{
        "article_id": "a", "chunk_id": "c", "citation_rank": 1,
        "title": "Báo cáo",
        "text": (
            "Doanh thu A năm 2024 là 10 tỷ đồng. "
            "Đoạn nhiễu hoàn toàn không liên quan. "
            "Doanh thu B năm 2024 là 20 tỷ đồng."
        ),
    }]
    plan = {"sub_questions": [
        {"id": "sq1", "text": "Doanh thu A năm 2024"},
        {"id": "sq2", "text": "Doanh thu B năm 2024"},
    ]}
    coverage = [
        {"sub_question_id": sub_id, "covered": True, "candidates": [{
            "article_id": "a", "chunk_id": "c", "supports": True,
        }]}
        for sub_id in ("sq1", "sq2")
    ]
    
    compressed, _ = compress_context_by_sentence(
        "So sánh doanh thu A và B", contexts, plan, coverage, threshold=0.99,
    )
    packed, stats = pack_contexts_with_budget(
        compressed, token_budget=200,
        route_decision={
            "selected_article_ids": ["a"],
            "covered_sub_questions": ["sq1", "sq2"],
        },
    )
    
    assert "10 tỷ" in packed[0]["text"] and "20 tỷ" in packed[0]["text"]
    assert stats["required_sub_questions_missing"] == []


def test_claim_prefix_is_stripped_from_verification_sub_question():
    plan = build_evidence_plan(
        'Nhận định sau đây đúng hay sai: "VN-Index tăng mạnh trong năm 2024"'
    )

    assert plan.answer_operator == "DIRECT"
    assert plan.sub_questions[0].evidence_type == "FACT"
    assert plan.sub_questions[0].text == (
        "Kiểm chứng thông tin: VN-Index tăng mạnh trong năm 2024"
    )


def test_explicit_article_topics_split_without_ranking_conclusion():
    plan = build_evidence_plan(
        "So sánh hai bài báo: bài về chính sách tiền tệ, "
        "bài về thị trường chứng khoán."
    )

    assert plan.answer_operator == "COMPARE"
    assert plan.estimated_sources_needed == 3
    assert [sq.text for sq in plan.sub_questions] == [
        "chính sách tiền tệ",
        "thị trường chứng khoán",
    ]
    assert [sq.evidence_type for sq in plan.sub_questions] == ["RELATION", "RELATION"]


def test_alphanumeric_entities_are_kept_intact_without_splitting():
    # Edge case: Ensure "Mazda 3" and "C180" do not lose their numeric parts due to regex constraints.
    plan = build_evidence_plan("Giá xe Mazda 3 và Mercedes-Benz C180 khác biệt ra sao?")
    
    assert plan.answer_operator == "COMPARE"
    assert "Mazda 3" in plan.entities
    assert "Mercedes-Benz C180" in plan.entities


def test_comparison_using_cua_keyword_extracts_correct_subjects():
    # Edge case: Comparisons using "của" (of) instead of "giữa" (between) still correctly extract the metric and both subjects.
    plan = build_evidence_plan("So sánh doanh thu của FPT và Hòa Phát trong năm 2023.")
    
    assert plan.answer_operator == "COMPARE"
    assert plan.estimated_sources_needed == 2
    assert "năm 2023" in plan.dates
    
    # Verified since _between_comparison_subjects now properly handles the keyword "của" or "trong".
    sub_texts = [sq.text for sq in plan.sub_questions]
    assert any("doanh thu FPT" in text for text in sub_texts)
    assert any("doanh thu Hòa Phát" in text for text in sub_texts)


def test_comparison_with_empty_metric_uses_generic_information_metric():
    # Edge case: If the user provides a bare comparison without a metric, fall back to "Thông tin" (Information).
    plan = build_evidence_plan("So sánh của FPT và Hòa Phát.")

    assert [sq.text for sq in plan.sub_questions] == [
        "Thông tin FPT",
        "Thông tin Hòa Phát",
    ]
    assert all(sq.evidence_type == "RELATION" for sq in plan.sub_questions)


def test_comparison_falls_back_to_entity_focus_when_no_between_pattern():
    plan = build_evidence_plan("So sánh FPT so với Hòa Phát về lợi nhuận")

    assert plan.answer_operator == "COMPARE"
    assert [sq.text for sq in plan.sub_questions] == [
        "FPT về lợi nhuận",
        "Hòa Phát về lợi nhuận",
    ]


def test_comparison_with_no_extractable_pair_keeps_original_question_as_relation():
    plan = build_evidence_plan("So sánh mức độ ảnh hưởng?")

    assert plan.answer_operator == "COMPARE"
    assert len(plan.sub_questions) == 1
    assert plan.sub_questions[0].text == "So sánh mức độ ảnh hưởng?"
    assert plan.sub_questions[0].evidence_type == "RELATION"


def test_comparison_operator_takes_precedence_over_timeline_and_causal_terms():
    plan = build_evidence_plan(
        "So sánh diễn biến và nguyên nhân tăng trưởng của FPT và Hòa Phát năm 2024"
    )

    assert plan.answer_operator == "COMPARE"
    assert plan.query_type_hint == "COMPARISON"
    assert plan.estimated_sources_needed == 2
    assert [sq.evidence_type for sq in plan.sub_questions] == ["RELATION", "RELATION"]


def test_multi_doc_comparison_detects_sources_accurately():
    # Edge case: Multi-document keywords like "trong các bài báo" safely elevate the requirement to 3 sources.
    plan = build_evidence_plan(
        "So sánh các bệnh nền được ghi nhận ở bệnh nhân mắc cúm A và sốt xuất huyết "
        "đang điều trị tại Bệnh viện Bệnh nhiệt đới Trung ương trong các bài báo."
    )
    
    assert plan.answer_operator == "COMPARE"
    assert plan.estimated_sources_needed == 3
    assert len(plan.sub_questions) >= 2


def test_causal_summary_with_complex_entities():
    # Edge case: Long causal/reasoning questions are summarized using 1 main source rather than triggering multi-source comparisons.
    plan = build_evidence_plan(
        "Vì sao Bí thư Tỉnh ủy Khánh Hòa lo ngại người dân có thể chủ quan trước bão Kalmaegi?"
    )
    
    assert plan.query_type_hint == "GENERAL"
    assert plan.answer_operator == "CAUSAL_SUMMARY"
    assert plan.estimated_sources_needed == 1
    assert plan.sub_questions[0].evidence_type == "CAUSAL"