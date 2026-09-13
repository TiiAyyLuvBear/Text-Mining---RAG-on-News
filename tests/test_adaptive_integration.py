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
