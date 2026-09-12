from src.RAG.retrieval.schema import GenerationDecision
from src.backend.generation_gate import build_retry_query, run_generation_stage
from src.backend.query_planner import build_evidence_plan
from src.backend.source_diversification import diversify_by_article
from src.backend.temporal_retrieval import apply_temporal_boost, temporal_terms_from_plan


def test_query_to_generation_contract_smoke():
    question = "So sánh doanh thu A và B năm 2025"
    plan = build_evidence_plan(question)
    candidates = [
        {
            "chunk_id": "ca",
            "article_id": "a",
            "title": "A năm 2025",
            "text": "Doanh thu A năm 2025 là 100 tỷ đồng.",
            "rerank_score": 0.90,
            "citation_rank": 1,
        },
        {
            "chunk_id": "cb",
            "article_id": "b",
            "title": "B năm 2025",
            "text": "Doanh thu B năm 2025 là 80 tỷ đồng.",
            "rerank_score": 0.89,
            "citation_rank": 2,
        },
    ]
    ranked = apply_temporal_boost(candidates, temporal_terms_from_plan(plan))
    selected = diversify_by_article(ranked, 1)
    coverage = [
        {
            "sub_question_id": "sq1",
            "covered": True,
            "covered_by_articles": ["a"],
            "candidates": [{"chunk_id": "ca", "article_id": "a", "supports": True}],
        },
        {
            "sub_question_id": "sq2",
            "covered": True,
            "covered_by_articles": ["b"],
            "candidates": [{"chunk_id": "cb", "article_id": "b", "supports": True}],
        },
    ]
    route = {
        "route": "REQUIRES_MULTI_DOC",
        "reason": "comparison needs both entities",
        "covered_sub_questions": ["sq1", "sq2"],
        "missing_sub_questions": [],
        "selected_article_ids": ["a", "b"],
        "retry_allowed": True,
    }

    result = run_generation_stage(
        question,
        plan,
        coverage,
        route,
        selected,
        generator_callback=lambda _question, _contexts: (
            "Doanh thu A năm 2025 là 100 tỷ đồng. [Nguồn 1]\n"
            "Doanh thu B năm 2025 là 80 tỷ đồng. [Nguồn 2]"
        ),
    )
    decision = GenerationDecision.model_validate(result)
    assert decision.decision == "ANSWER"
    assert decision.verification_status == "PASS"
    assert [citation.citation_rank for citation in decision.citations] == [1, 2]


def test_multi_entity_question_can_still_route_single_document():
    question = "Vai trò của A và B trong sự kiện X là gì?"
    plan = build_evidence_plan(question)
    assert plan.estimated_sources_needed == 1
    candidate = {
        "chunk_id": "cab",
        "article_id": "ab",
        "text": "A tổ chức sự kiện X, còn B tài trợ sự kiện X.",
        "rerank_score": 0.9,
        "citation_rank": 1,
    }
    result = run_generation_stage(
        question,
        plan,
        [{
            "sub_question_id": "sq1",
            "covered": True,
            "covered_by_articles": ["ab"],
            "candidates": [{
                "chunk_id": "cab",
                "article_id": "ab",
                "supports": True,
            }],
        }],
        {
            "route": "SINGLE_DOC",
            "reason": "one article covers the full question",
            "covered_sub_questions": ["sq1"],
            "missing_sub_questions": [],
            "selected_article_ids": ["ab"],
            "retry_allowed": True,
        },
        [candidate],
        generator_callback=lambda _question, _contexts: (
            "A tổ chức sự kiện X, còn B tài trợ sự kiện X. [Nguồn 1]"
        ),
    )
    assert GenerationDecision.model_validate(result).decision == "ANSWER"


def test_retry_query_can_be_replanned_upstream():
    plan = build_evidence_plan("So sánh doanh thu A và B năm 2025")
    retry_query = build_retry_query(
        plan.normalized_question,
        plan,
        {"missing_sub_questions": ["sq2"]},
    )
    retry_plan = build_evidence_plan(retry_query)
    assert retry_query == "doanh thu B năm 2025"
    assert retry_plan.normalized_question == retry_query
    assert retry_plan.dates == ["năm 2025"]
