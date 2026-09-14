import io
import json

from src.RAG.retrieval.schema import GenerationDecision
from src.backend.evidence_router import (
    INSUFFICIENT,
    REQUIRES_MULTI_DOC,
    SINGLE_DOC,
    route_evidence,
)


PLAN = {
    "normalized_question": "So sánh A và B",
    "estimated_sources_needed": 2,
    "sub_questions": [{"id": "sq1", "text": "A"}, {"id": "sq2", "text": "B"}],
}


def routed(mapping):
    candidates = [
        {"article_id": article, "chunk_id": f"c-{article}", "text": article}
        for article in sorted({article for article, _ in mapping})
    ]
    return route_evidence(
        PLAN,
        candidates,
        support_scorer=lambda sub, candidate: 1.0 if (candidate["article_id"], sub) in mapping else 0.0,
    )


def test_coverage_route_is_not_planner_source_count():
    result = routed({("article-a", "A"), ("article-a", "B")})
    assert result["route_decision"]["route"] == SINGLE_DOC
    assert result["route_decision"]["selected_article_ids"] == ["article-a"]


def test_equal_minimal_covers_preserve_evidence_rank_not_article_id_sort():
    plan = {"sub_questions": [{"id": "sq1", "text": "purin và thận"}]}
    candidates = [
        {"article_id": "211640", "chunk_id": "211640_1", "text": "strong"},
        {"article_id": "117558", "chunk_id": "117558_1", "text": "weak"},
    ]

    result = route_evidence(plan, candidates, support_scorer=lambda *_: 1.0)

    assert result["route_decision"]["route"] == SINGLE_DOC
    assert result["route_decision"]["selected_article_ids"] == ["211640"]


def test_coverage_route_requires_two_articles_only_when_needed():
    result = routed({("article-a", "A"), ("article-b", "B")})
    assert result["route_decision"]["route"] == REQUIRES_MULTI_DOC
    assert result["route_decision"]["selected_article_ids"] == ["article-a", "article-b"]


def test_coverage_route_refuses_missing_requirement():
    result = routed({("article-a", "A")})
    assert result["route_decision"]["route"] == INSUFFICIENT
    assert result["route_decision"]["missing_sub_questions"] == ["sq2"]


def test_multiple_entities_can_use_one_article():
    result = routed({("article-ab", "A"), ("article-ab", "B")})
    assert result["route_decision"]["route"] == SINGLE_DOC


def test_lexical_coverage_does_not_confuse_named_entities():
    plan = {"sub_questions": [{"id": "sq1", "text": "Doanh thu A năm 2025"}]}
    result = route_evidence(plan, [{"article_id": "b", "chunk_id": "c", "text": "Doanh thu B năm 2025 là 100."}])
    assert result["route_decision"]["route"] == INSUFFICIENT


def test_sentence_initial_question_word_is_not_a_proper_name_anchor():
    plan = {"sub_questions": [{
        "id": "sq1",
        "text": "Những loại nội tạng nào làm tăng axit uric và hại thận?",
    }]}
    result = route_evidence(plan, [{
        "article_id": "211640",
        "chunk_id": "211640_1",
        "text": "Nội tạng chứa purin; purin làm tăng axit uric và ảnh hưởng đến thận.",
    }])

    assert result["route_decision"]["route"] == SINGLE_DOC
    assert result["route_decision"]["selected_article_ids"] == ["211640"]


def test_single_digit_and_apostrophe_name_do_not_create_impossible_anchors():
    cases = [
        (
            "Vì sao khu đất số 5 Lê Lợi gây bất ngờ?",
            "Khu đất số 5 Lê Lợi gây bất ngờ vì doanh nghiệp trúng đấu giá còn non trẻ.",
        ),
        (
            "Vì sao H'Hen Niê đảm nhận vị trí vedette?",
            "Hoa hậu H'Hen Niê đảm nhận vị trí vedette của bộ sưu tập Nguyệt Hồi.",
        ),
    ]
    for question, text in cases:
        result = route_evidence(
            {"sub_questions": [{"id": "sq1", "text": question}]},
            [{"article_id": "gold", "chunk_id": "c", "text": text}],
        )
        assert result["route_decision"]["route"] == SINGLE_DOC


def test_model_identifier_is_matched_as_a_whole_not_as_numeric_suffix():
    plan = {"sub_questions": [{
        "id": "sq1",
        "text": "Mức giá Mercedes-Benz C180 cũ ngang bằng mẫu xe nào?",
    }]}
    matching = route_evidence(plan, [{
        "article_id": "gold", "chunk_id": "c1",
        "title": "Mercedes-Benz C180 cũ",
        "text": "Mức giá hiện tại ngang bằng một mẫu xe tay ga Honda Air Blade.",
    }])
    wrong_model = route_evidence(plan, [{
        "article_id": "wrong", "chunk_id": "c2",
        "title": "Mercedes-Benz C200 cũ",
        "text": "Mức giá hiện tại ngang bằng một mẫu xe tay ga Honda Air Blade.",
    }])
    assert matching["route_decision"]["route"] == SINGLE_DOC
    assert wrong_model["route_decision"]["route"] == INSUFFICIENT


def test_vietnamese_organisation_aliases_are_controlled_equivalents():
    plan = {"sub_questions": [{
        "id": "sq1", "text": "Bộ GD-ĐT công bố lịch tuyển sinh khi nào?",
    }]}
    result = route_evidence(plan, [{
        "article_id": "education", "chunk_id": "c",
        "text": "Bộ Giáo dục và Đào tạo công bố lịch tuyển sinh vào tháng 7.",
    }])
    assert result["route_decision"]["route"] == SINGLE_DOC
    candidate = result["coverage_matrix"][0]["best_candidate"]
    assert candidate["entity_score"] == 1.0


def test_coverage_diagnostics_explain_missing_subquestion():
    result = route_evidence(
        {"sub_questions": [{"id": "sq1", "text": "Doanh thu A năm 2025"}]},
        [{"article_id": "b", "chunk_id": "c", "text": "Doanh thu B năm 2025 là 100."}],
    )
    row = result["coverage_matrix"][0]
    assert row["covered"] is False
    assert row["best_candidate"]["support_score"] == 0.0
    assert row["failure_reason"] in {
        "entity_mismatch", "temporal_mismatch", "support_score_below_threshold",
    }
    assert set((
        "lexical_score", "entity_score", "concept_score", "temporal_score", "relation_score",
    )) <= row["best_candidate"].keys()


def test_long_unrelated_chunk_does_not_gain_support_from_global_keyword_scatter():
    filler = " ".join(
        ["Doanh thu được nhắc ở một chủ đề khác."] * 20
        + ["Công ty A xuất hiện trong phần tiểu sử."] * 20
        + ["Năm 2025 là mốc xuất bản của bài viết."] * 20
    )
    result = route_evidence(
        {"sub_questions": [{"id": "sq1", "text": "Doanh thu A năm 2025"}]},
        [{"article_id": "noise", "chunk_id": "c", "text": filler}],
    )
    assert result["route_decision"]["route"] == INSUFFICIENT


def test_unanswerable_relation_is_not_supported_by_topic_only_evidence():
    cases = [
        (
            "Mazda 3e dự kiến sẽ được bán với mức giá bao nhiêu tại từng thị trường?",
            "Mazda đã đăng ký tên Mazda 3e tại Úc, Anh và châu Âu.",
        ),
        (
            "Hoa hậu H'Hen Niê cảm nhận thế nào khi được mời làm vedette Nguyệt Hồi?",
            "H'Hen Niê làm vedette cho bộ sưu tập Nguyệt Hồi.",
        ),
        (
            "Bão Kalmaegi gây thiệt hại cụ thể thế nào tại Khánh Hòa?",
            "Khánh Hòa sẵn sàng sơ tán để phòng tránh thiệt hại do bão Kalmaegi.",
        ),
    ]
    for question, text in cases:
        from src.backend.query_planner import build_evidence_plan
        result = route_evidence(
            build_evidence_plan(question).model_dump(),
            [{"article_id": "topic", "chunk_id": "c", "text": text}],
        )
        assert result["route_decision"]["route"] == INSUFFICIENT
        assert result["coverage_matrix"][0]["failure_reason"] == "relation_mismatch"


def test_real_temporal_comparison_routes_from_collective_evidence():
    from src.backend.query_planner import build_evidence_plan

    question = (
        "So sánh thời gian đăng ký xét tuyển trên Cổng thông tin tuyển sinh của "
        "Bộ GD-ĐT giữa Trường ĐH Công nghệ Giao thông vận tải năm 2024 và "
        "Trường ĐH Ngoại thương năm 2025. Có sự khác biệt nào về thời hạn đăng ký?"
    )
    plan = build_evidence_plan(question).model_dump()
    candidates = [
        {
            "article_id": "178937", "chunk_id": "178937_token_0000",
            "title": "Điểm sàn Trường ĐH Công nghệ Giao thông vận tải năm 2024",
            "text": "Đăng ký xét tuyển trên Hệ thống Bộ GD-ĐT từ ngày 18/7 đến 17h ngày 30/7.",
        },
        {
            "article_id": "177700", "chunk_id": "177700_token_0000",
            "title": "Điểm sàn Trường ĐH Ngoại thương năm 2025",
            "text": "Đăng ký trên Cổng tuyển sinh Bộ GD-ĐT từ ngày 16/7 đến 17h ngày 28/7.",
        },
    ]

    result = route_evidence(plan, candidates)
    matrix = result["coverage_matrix"]

    assert all(item["covered"] for item in matrix)
    assert not any(
        all(article_id in item["covered_by_articles"] for item in matrix)
        for article_id in {"178937", "177700"}
    )
    assert result["route_decision"]["route"] == REQUIRES_MULTI_DOC
    assert set(result["route_decision"]["selected_article_ids"]) == {"178937", "177700"}


def test_related_articles_do_not_answer_an_unsupported_subjective_ranking():
    from src.backend.query_planner import build_evidence_plan

    question = (
        "So sánh quan điểm của ba bài báo: bài về tuổi gia chủ, bài về ngày giờ "
        "động thổ, và bài về kiêng kỵ thiết kế nhà. Yếu tố nào gây hậu quả "
        "nghiêm trọng nhất nếu vi phạm?"
    )
    plan = build_evidence_plan(question).model_dump()
    candidates = [
        {"article_id": "150854", "chunk_id": "a", "text": "Chọn ngày xấu gây trục trặc và thiệt hại tiền bạc."},
        {"article_id": "150858", "chunk_id": "b", "text": "Tuổi gia chủ phạm hạn có thể gây khó khăn và bệnh tật."},
        {"article_id": "150614", "chunk_id": "c", "text": "Kiêng kỵ thiết kế nhà ảnh hưởng tài lộc và sức khỏe."},
    ]

    result = route_evidence(plan, candidates)

    assert result["route_decision"]["route"] == INSUFFICIENT
    conclusion = next(
        item for item in result["coverage_matrix"]
        if item["sub_question_id"] == plan["sub_questions"][-1]["id"]
    )
    assert conclusion["covered"] is False


def test_adaptive_retry_uses_refreshed_candidates_and_route(monkeypatch):
    from src.backend.pipeline import NewsPipeline

    pipeline = NewsPipeline.__new__(NewsPipeline)
    plan = {"normalized_question": "A?", "sub_questions": [{"id": "sq1", "text": "A?"}]}
    stale = {"article_id": "stale", "chunk_id": "old", "text": "không liên quan", "rerank_score": 1.0}
    fresh = {"article_id": "fresh", "chunk_id": "new", "text": "A là 100.", "rerank_score": 1.0, "citation_rank": 4}
    states = [
        (plan, [stale], [{"sub_question_id": "sq1", "candidates": [], "covered": False}], {"route": "INSUFFICIENT", "missing_sub_questions": ["sq1"], "selected_article_ids": [], "retry_allowed": True}),
        (plan, [fresh], [{"sub_question_id": "sq1", "candidates": [{"article_id": "fresh", "chunk_id": "new", "supports": True}], "covered": True, "covered_by_articles": ["fresh"]}], {"route": "SINGLE_DOC", "covered_sub_questions": ["sq1"], "missing_sub_questions": [], "selected_article_ids": ["fresh"], "retry_allowed": True}),
    ]
    calls = []

    def prepare(question, evidence_plan=None):
        calls.append((question, evidence_plan))
        return states.pop(0)

    pipeline._adaptive_evidence = prepare
    pipeline.generate = lambda question, contexts: "A là 100. [Nguồn 4]" if contexts[0]["article_id"] == "fresh" else "bad"
    result = pipeline.search_adaptive("A?")
    assert result.decision == "ANSWER"
    assert result.retry_count == 1
    assert len(calls) == 2 and calls[1][1] == plan
    assert result.citations[0].citation_rank == 4


def test_failed_retry_refuses_without_calling_generator():
    from src.backend.pipeline import NewsPipeline

    pipeline = NewsPipeline.__new__(NewsPipeline)
    plan = {"normalized_question": "Thiếu gì?", "sub_questions": [{"id": "sq1", "text": "Thiếu gì?"}]}
    state = (
        plan,
        [{"article_id": "noise", "chunk_id": "c", "text": "không liên quan"}],
        [{"sub_question_id": "sq1", "covered": False, "candidates": []}],
        {
            "route": "INSUFFICIENT", "missing_sub_questions": ["sq1"],
            "selected_article_ids": [], "retry_allowed": True,
        },
    )
    calls = []
    pipeline._adaptive_evidence = lambda *args, **kwargs: state
    pipeline.generate = lambda *args, **kwargs: calls.append(1) or "bad"

    result = pipeline.search_adaptive("Thiếu gì?")

    assert result.decision == "REFUSE"
    assert result.retry_count == 1
    assert calls == []


def _answer():
    return GenerationDecision(
        decision="ANSWER", answer="Purin hòa tan. [Nguồn 1]",
        citations=[], verification_status="PASS",
    )


def test_rest_and_websocket_use_search_adaptive(monkeypatch):
    import src.backend.pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module, "QdrantClient", lambda path: object())
    import src.backend.app as app

    class Pipeline:
        last_ranked_contexts = [{"article_id": "a", "chunk_id": "c", "text": "Purin hòa tan.", "citation_rank": 1}]
        def search_adaptive(self, question, top_k): return _answer()
        def close(self): pass

    monkeypatch.setattr(app, "pipeline", Pipeline())
    payload = app.ask(app.AskRequest(question="purin", top_k=1))
    assert payload["decision"] == "ANSWER"


def test_rest_generator_error_keeps_evidence_status_and_skips_evaluation(monkeypatch):
    import src.backend.pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module, "QdrantClient", lambda path: object())
    import src.backend.app as app

    class Pipeline:
        last_ranked_contexts = [{"article_id": "a", "chunk_id": "c", "text": "evidence"}]
        last_planned_contexts = [{"article_id": "a", "chunk_id": "c", "text": "evidence", "citation_rank": 1}]
        last_evidence_plan = {"sub_questions": [{"id": "sq1", "text": "purin"}]}
        last_coverage_matrix = [{"sub_question_id": "sq1", "covered": True}]
        last_route_decision = {"route": "SINGLE_DOC", "selected_article_ids": ["a"]}
        def search_adaptive(self, question, top_k):
            return GenerationDecision(
                decision="REFUSE", answer="", refusal_reason="generation_unavailable",
                refusal_reason_code="GENERATOR_ERROR", failure_category="GENERATOR",
                verification_status="NOT_RUN",
            )
        def close(self): pass

    monkeypatch.setattr(app, "pipeline", Pipeline())
    payload = app.ask(app.AskRequest(question="purin", top_k=1))

    assert payload["answer_status"] == "generation_unavailable"
    assert payload["evidence_sufficient"] is True
    assert payload["evaluation"]["status"] == "skipped"
    assert payload["verification_status"] == "NOT_RUN"
    assert payload["evidence_plan"]["sub_questions"][0]["id"] == "sq1"
    assert payload["coverage_matrix"][0]["covered"] is True


def test_websocket_uses_search_adaptive(monkeypatch):
    import src.backend.pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module, "QdrantClient", lambda path: object())
    import src.backend.app as app

    class Pipeline:
        last_ranked_contexts = [{"article_id": "a", "chunk_id": "c", "text": "Purin hòa tan.", "citation_rank": 1}]
        def search_adaptive(self, question, top_k): return _answer()
        def close(self): pass

    monkeypatch.setattr(app, "pipeline", Pipeline())
    from fastapi.testclient import TestClient
    with TestClient(app.app) as client:
        with client.websocket_connect("/api/qa/stream") as socket:
            socket.send_json({"question": "purin"})
            assert "Purin" in socket.receive_text()


def test_legacy_http_uses_search_adaptive(monkeypatch):
    import src.backend.pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module, "QdrantClient", lambda path: object())
    import src.backend.legacy_http as legacy

    class Pipeline:
        last_ranked_contexts = [{"article_id": "a", "chunk_id": "c", "text": "Purin hòa tan.", "citation_rank": 1}]
        def search_adaptive(self, question, top_k): return _answer()

    monkeypatch.setattr(legacy, "PIPELINE", Pipeline())

    class Handler(legacy.RagHandler):
        def __init__(self, body):
            self.headers = {"Content-Length": str(len(body))}; self.path = "/ask"
            self.rfile = io.BytesIO(body); self.wfile = io.BytesIO()
        def send_response(self, status): self.status = status
        def send_header(self, *args): pass
        def end_headers(self): pass

    handler = Handler(json.dumps({"question": "purin"}).encode())
    handler.do_POST()
    assert handler.status == 200
    assert json.loads(handler.wfile.getvalue())["decision"] == "ANSWER"
