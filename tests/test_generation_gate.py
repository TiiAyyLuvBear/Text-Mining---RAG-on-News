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


def coverage(*pairs):
    return [
        {
            "sub_question_id": sub_id,
            "covered": True,
            "covered_by_articles": [article_id],
            "candidates": [{
                "article_id": article_id,
                "chunk_id": f"c-{article_id}",
                "support_score": 1.0,
                "supports": True,
            }],
        }
        for sub_id, article_id in pairs
    ]


def test_single_doc_uses_only_selected_article():
    seen = {}

    def generator(question, contexts):
        seen["question"] = question
        seen["articles"] = [item["article_id"] for item in contexts]
        return "Doanh thu A năm 2025 là 100 tỷ đồng. [Nguồn 1]"

    result = run_generation_stage(
        "Doanh thu A năm 2025 là bao nhiêu?",
        PLAN,
        coverage(("sq1", "a")),
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
        coverage(("sq1", "a"), ("sq2", "b")),
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
            "coverage_matrix": coverage(("sq2", "b")),
            "route_decision": {"route": "SINGLE_DOC", "selected_article_ids": ["b"], "covered_sub_questions": ["sq2"]},
        }

    original = "So sánh doanh thu A và B năm 2025"
    result = run_generation_stage(
        original,
        PLAN,
        coverage(("sq1", "a")),
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
        coverage(("sq1", "a")),
        {"route": "SINGLE_DOC", "selected_article_ids": ["a"], "covered_sub_questions": ["sq1"]},
        [candidate("a", "Doanh thu A là 100 tỷ đồng.", 1)],
        generator_callback=lambda question, contexts: "Doanh thu A là 900 tỷ đồng. [Nguồn 1]",
    )
    assert result["decision"] == "REFUSE"
    assert result["verification_status"] == "FAIL"
    assert result["refusal_reason"] == "generation_verification_failed"
    assert result["failure_category"] == "VERIFICATION"
    assert result["missing_evidence"] == []


def test_build_retry_query_prefers_missing_subquestion():
    query = build_retry_query(
        "So sánh doanh thu A và B năm 2025",
        PLAN,
        {"missing_sub_questions": ["sq2"]},
    )
    assert query.startswith("Doanh thu B năm 2025")
    assert "bao nhiêu" not in query.lower()
    assert not query.endswith(" A")


def test_single_doc_selects_article_that_covers_all_not_first_id():
    seen = []

    def generator(question, contexts):
        seen.extend(item["article_id"] for item in contexts)
        return "Doanh thu A đạt 100 tỷ đồng. [Nguồn 2]"

    result = run_generation_stage(
        "Doanh thu A?",
        PLAN,
        coverage(("sq1", "good")),
        {
            "route": "SINGLE_DOC",
            "selected_article_ids": ["wrong", "good"],
            "covered_sub_questions": ["sq1"],
        },
        [
            candidate("wrong", "Tin không đủ.", 1, score=9.0),
            candidate("good", "Doanh thu A đạt 100 tỷ đồng.", 2, score=2.0),
        ],
        generator_callback=generator,
    )
    assert result["decision"] == "ANSWER"
    assert seen == ["good"]


def test_selected_article_missing_from_candidates_is_contract_refusal():
    result = run_generation_stage(
        "Doanh thu A?",
        PLAN,
        coverage(("sq1", "missing")),
        {"route": "SINGLE_DOC", "selected_article_ids": ["missing"], "covered_sub_questions": ["sq1"]},
        [candidate("a", "Doanh thu A đạt 100 tỷ đồng.", 1)],
        generator_callback=lambda question, contexts: "must not run",
    )
    assert result["decision"] == "REFUSE"
    assert result["failure_category"] == "CONTRACT"
    assert result["verification_status"] == "NOT_RUN"


def test_malformed_retry_route_decision_is_rejected():
    result = run_generation_stage(
        "Câu hỏi",
        PLAN,
        [],
        {"route": "INSUFFICIENT", "missing_sub_questions": ["sq2"], "retry_allowed": True},
        [],
        retry_callback=lambda query: {
            "ranked_candidates": [],
            "coverage_matrix": [],
            "route_decision": {"route": "BROKEN"},
        },
        generator_callback=lambda question, contexts: "must not run",
    )
    assert result["refusal_reason_code"] == "INVALID_RETRY_RESULT"
    assert result["retry_count"] == 1


def test_retry_callback_exception_is_contract_failure_not_crash():
    result = run_generation_stage(
        "Câu hỏi",
        PLAN,
        [],
        {"route": "INSUFFICIENT", "missing_sub_questions": ["sq2"], "retry_allowed": True},
        [],
        retry_callback=lambda query: (_ for _ in ()).throw(TimeoutError("upstream timeout")),
        generator_callback=lambda question, contexts: "must not run",
    )
    assert result["decision"] == "REFUSE"
    assert result["refusal_reason_code"] == "RETRY_CALLBACK_ERROR"
    assert result["failure_category"] == "CONTRACT"
    assert result["retry_count"] == 1


def test_tiny_multidoc_budget_blocks_before_generator():
    called = []
    result = run_generation_stage(
        "So sánh A và B",
        PLAN,
        coverage(("sq1", "a"), ("sq2", "b")),
        {
            "route": "REQUIRES_MULTI_DOC",
            "selected_article_ids": ["a", "b"],
            "covered_sub_questions": ["sq1", "sq2"],
        },
        [
            candidate("a", "Doanh thu A là một trăm tỷ đồng.", 1, score=99.0),
            candidate("b", "Doanh thu B là tám mươi tỷ đồng.", 2),
        ],
        generator_callback=lambda question, contexts: called.append(contexts),
        token_counter=lambda text: len(text.split()),
        token_budget=5,
        compression_enabled=True,
    )
    assert result["decision"] == "REFUSE"
    assert result["refusal_reason_code"] == "CONTEXT_BUDGET_INSUFFICIENT"
    assert result["failure_category"] == "CONTEXT_BUDGET"
    assert called == []


def test_generator_exception_has_distinct_generator_category():
    result = run_generation_stage(
        "Doanh thu A?",
        PLAN,
        coverage(("sq1", "a")),
        {"route": "SINGLE_DOC", "selected_article_ids": ["a"], "covered_sub_questions": ["sq1"]},
        [candidate("a", "Doanh thu A đạt 100 tỷ đồng.", 1)],
        generator_callback=lambda question, contexts: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    assert result["decision"] == "REFUSE"
    assert result["refusal_reason_code"] == "GENERATOR_ERROR"
    assert result["failure_category"] == "GENERATOR"
    assert result["verification_status"] == "NOT_RUN"


def test_non_string_generator_response_is_parse_failure():
    result = run_generation_stage(
        "Doanh thu A?",
        PLAN,
        coverage(("sq1", "a")),
        {"route": "SINGLE_DOC", "selected_article_ids": ["a"], "covered_sub_questions": ["sq1"]},
        [candidate("a", "Doanh thu A đạt 100 tỷ đồng.", 1)],
        generator_callback=lambda question, contexts: {"answer": "unexpected shape"},
    )
    assert result["decision"] == "REFUSE"
    assert result["refusal_reason_code"] == "GENERATOR_RESPONSE_INVALID"
    assert result["failure_category"] == "PARSE"
    assert result["verification_status"] == "NOT_RUN"


def test_sentence_pruning_can_be_disabled_for_safe_integration():
    seen = {}

    def generator(question, contexts):
        seen["text"] = contexts[0]["text"]
        return "Doanh thu A đạt 100 tỷ đồng. [Nguồn 1]"

    result = run_generation_stage(
        "Doanh thu A?",
        PLAN,
        coverage(("sq1", "a")),
        {"route": "SINGLE_DOC", "selected_article_ids": ["a"], "covered_sub_questions": ["sq1"]},
        [candidate("a", "Câu hoàn toàn không liên quan. Doanh thu A đạt 100 tỷ đồng.", 1)],
        generator_callback=generator,
        compression_enabled=False,
    )
    assert result["decision"] == "ANSWER"
    assert "Câu hoàn toàn không liên quan." in seen["text"]
    assert result["compression_stats"]["compression_enabled"] is False


def test_accepts_actual_pydantic_upstream_contracts():
    from src.RAG.retrieval.schema import (
        CandidateSupport,
        CoverageMatrix,
        EvidencePlan,
        RouteDecision,
        SubQuestion,
    )

    plan = EvidencePlan(
        normalized_question="Doanh thu A?",
        query_type_hint="FACTOID",
        answer_operator="DIRECT",
        sub_questions=[SubQuestion(id="sq1", text="Doanh thu A?", evidence_type="FACT")],
    )
    matrix = [CoverageMatrix(
        sub_question_id="sq1",
        covered=True,
        covered_by_articles=["a"],
        candidates=[CandidateSupport(chunk_id="c-a", article_id="a", support_score=1.0, supports=True)],
    )]
    route = RouteDecision(
        route="SINGLE_DOC",
        reason="one article covers all",
        covered_sub_questions=["sq1"],
        selected_article_ids=["a"],
    )
    result = run_generation_stage(
        "Doanh thu A?",
        plan,
        matrix,
        route,
        [candidate("a", "Doanh thu A đạt 100 tỷ đồng.", 1)],
        generator_callback=lambda question, contexts: "Doanh thu A đạt 100 tỷ đồng. [Nguồn 1]",
    )
    assert result["decision"] == "ANSWER"
