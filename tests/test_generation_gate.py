from src.backend.generation_gate import (
    build_retry_query,
    run_generation_stage,
)


PLAN = {
    "normalized_question": "So sánh doanh thu A và B năm 2025",
    "entities": ["A", "B"],
    "numbers": [],
    "dates": ["2025"],
    "temporal_constraints": [],
    "answer_operator": "COMPARE",
    "sub_questions": [
        {"id": "sq1", "text": "Doanh thu A năm 2025 là bao nhiêu?", "evidence_type": "FACT"},
        {"id": "sq2", "text": "Doanh thu B năm 2025 là bao nhiêu?", "evidence_type": "FACT"},
    ],
}


def candidate(article_id, text, rank, chunk_id=None, score=2.0):
    return {
        "article_id": article_id,
        "chunk_id": chunk_id or f"c-{article_id}",
        "citation_rank": rank,
        "title": f"Tin {article_id}",
        "text": text,
        "rerank_score": score,
    }


def test_single_doc_uses_only_selected_article():
    seen = {}

    def generator(question, contexts):
        seen["question"] = question
        seen["articles"] = [item["article_id"] for item in contexts]
        return "Doanh thu A năm 2025 là 100 tỷ đồng. [Nguồn 1]"

    result = run_generation_stage(
        "Doanh thu A năm 2025 là bao nhiêu?",
        PLAN,
        [],
        {"route": "SINGLE_DOC", "selected_article_ids": ["a"], "covered_sub_questions": ["sq1"]},
        [
            candidate("a", "Doanh thu A năm 2025 là 100 tỷ đồng.", 1),
            candidate("b", "Doanh thu B năm 2025 là 80 tỷ đồng.", 2),
        ],
        generator_callback=generator,
    )
    assert result["decision"] == "ANSWER"
    assert seen["articles"] == ["a"]


def test_multidoc_uses_all_selected_sources():
    seen = {}

    def generator(question, contexts):
        seen["articles"] = {item["article_id"] for item in contexts}
        return "Doanh thu A là 100 tỷ đồng. [Nguồn 1]\nDoanh thu B là 80 tỷ đồng. [Nguồn 2]"

    result = run_generation_stage(
        "So sánh doanh thu A và B",
        PLAN,
        [],
        {"route": "REQUIRES_MULTI_DOC", "selected_article_ids": ["a", "b"], "covered_sub_questions": ["sq1", "sq2"]},
        [
            candidate("a", "Doanh thu A là 100 tỷ đồng.", 1),
            candidate("b", "Doanh thu B là 80 tỷ đồng.", 2),
        ],
        generator_callback=generator,
    )
    assert result["decision"] == "ANSWER"
    assert seen["articles"] == {"a", "b"}
    assert len(result["citations"]) == 2


def test_insufficient_retries_once_with_separate_focused_query():
    calls = []

    def retry(query):
        calls.append(query)
        return {
            "ranked_candidates": [candidate("b", "Doanh thu B năm 2025 là 80 tỷ đồng.", 2)],
            "coverage_matrix": [],
            "route_decision": {"route": "SINGLE_DOC", "selected_article_ids": ["b"], "covered_sub_questions": ["sq2"]},
        }

    original = "So sánh doanh thu A và B năm 2025"
    result = run_generation_stage(
        original,
        PLAN,
        [],
        {"route": "INSUFFICIENT", "missing_sub_questions": ["sq2"], "retry_allowed": True},
        [],
        retry_callback=retry,
        generator_callback=lambda question, contexts: "Doanh thu B năm 2025 là 80 tỷ đồng. [Nguồn 2]",
    )
    assert len(calls) == 1
    assert calls[0] != original
    assert "Doanh thu B năm 2025" in calls[0]
    assert result["retry_count"] == 1
    assert result["decision"] == "ANSWER"


def test_retry_disabled_refuses_without_generation():
    called = []
    result = run_generation_stage(
        "Câu hỏi",
        PLAN,
        [],
        {"route": "INSUFFICIENT", "missing_sub_questions": ["sq2"], "retry_allowed": False},
        [],
        retry_callback=lambda query: called.append(query),
        generator_callback=lambda question, contexts: (_ for _ in ()).throw(AssertionError("must not generate")),
    )
    assert result["decision"] == "REFUSE"
    assert result["missing_evidence"] == ["sq2"]
    assert called == []


def test_retry_still_insufficient_refuses_and_does_not_retry_twice():
    calls = []

    def retry(query):
        calls.append(query)
        return {
            "ranked_candidates": [],
            "coverage_matrix": [],
            "route_decision": {"route": "INSUFFICIENT", "missing_sub_questions": ["sq2"], "retry_allowed": True},
        }

    result = run_generation_stage(
        "Câu hỏi",
        PLAN,
        [],
        {"route": "INSUFFICIENT", "missing_sub_questions": ["sq2"], "retry_allowed": True},
        [],
        retry_callback=retry,
        generator_callback=lambda question, contexts: "must not run",
    )
    assert len(calls) == 1
    assert result["decision"] == "REFUSE"
    assert result["refusal_reason"] == "evidence_insufficient_after_retry"
    assert result["retry_count"] == 1


def test_verification_failure_is_not_reported_as_missing_evidence():
    result = run_generation_stage(
        "Doanh thu A?",
        PLAN,
        [],
        {"route": "SINGLE_DOC", "selected_article_ids": ["a"], "covered_sub_questions": ["sq1"]},
        [candidate("a", "Doanh thu A là 100 tỷ đồng.", 1)],
        generator_callback=lambda question, contexts: "Doanh thu A là 900 tỷ đồng.",
    )
    assert result["decision"] == "REFUSE"
    assert result["verification_status"] == "FAIL"
    assert result["refusal_reason"] == "generation_verification_failed"
    assert result["missing_evidence"] == []


def test_build_retry_query_prefers_missing_subquestion():
    query = build_retry_query(
        "So sánh doanh thu A và B năm 2025",
        PLAN,
        {"missing_sub_questions": ["sq2"]},
    )
    assert query.startswith("Doanh thu B năm 2025")
    assert "bao nhiêu" not in query.lower()
