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
