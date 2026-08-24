from src.backend.evaluation import claim_support, evaluate_response, source_diversity


def contexts():
    return [
        {"article_id": "a", "text": "Purin hòa vào nước khi hầm lâu.", "title": "Purin"},
        {"article_id": "b", "text": "Uống đủ nước hỗ trợ cơ thể.", "title": "Nước"},
    ]


def test_groundedness_marks_unsupported_claim():
    result = claim_support(
        "Purin hòa vào nước khi hầm lâu. Nước hầm chắc chắn gây suy thận.", contexts()
    )
    assert result["claim_count"] == 2
    assert result["supported_claims"] == 1
    assert result["lexical_support_coverage"] == 0.5


def test_source_diversity_deduplicates_articles():
    result = source_diversity(contexts() + [{"article_id": "a", "text": "x"}])
    assert result["unique_articles"] == 2
    assert result["ratio"] == round(2 / 3, 4)


def test_evaluation_recommends_abstention_for_unsupported_answer():
    result = evaluate_response("purin", "Sấm sét gây bệnh.", contexts(), True)
    assert result["abstention_recommended"] is True
    assert result["confidence_semantics"].startswith("not calibrated")

def test_negation_is_not_removed_and_unknown_empty_abstains():
    result = claim_support("Purin không gây bệnh.", [{"text": "Purin gây bệnh."}])
    assert result["claim_count"] == 1
    assert result["contradicted_claims"] == 1
    assert evaluate_response("purin", "", contexts(), True)["abstention_recommended"] is True

def test_citation_validation_range_and_detached_marker():
    result = claim_support("Theo nguồn: purin hòa vào nước [Nguồn 99].", contexts())
    assert result["citation_errors"][0]["type"] == "out_of_range"

def test_unicode_malformed_context_safe():
    result = evaluate_response("\ud800 purin", "Purin.", [{"text": None}], False)
    assert result["abstention_recommended"] is True


def test_pipeline_selection_renumbers_and_gate_uses_full_pool(monkeypatch):
    from src.backend.pipeline import NewsPipeline
    pipeline = NewsPipeline.__new__(NewsPipeline)
    ranked = [{"article_id": "a", "rank": 1, "chunk_id": "1", "text": "x", "rerank_score": 5.0},
              {"article_id": "a", "rank": 2, "chunk_id": "2", "text": "y", "rerank_score": 4.9},
              {"article_id": "b", "rank": 3, "chunk_id": "3", "text": "z", "rerank_score": 4.0}]
    monkeypatch.setattr(pipeline, "retrieve", lambda question: ranked)
    monkeypatch.setattr(pipeline, "rerank", lambda question, candidates: candidates)
    selected, sufficient, top, margin = pipeline.search_with_evidence("q", 2)
    assert [item["article_id"] for item in selected] == ["a", "b"]
    assert [item["rank"] for item in selected] == [1, 3]
    assert [item["citation_rank"] for item in selected] == [1, 2]
    assert sufficient is False
    assert top == 5.0 and margin == 1.0

def test_citation_support_checks_cited_source_not_best_source():
    result = claim_support("Purin hòa vào nước [Nguồn 2].", contexts())
    assert result["claims"][0]["citation_presence"] is True
    assert result["claims"][0]["citation_support"] is False
    assert result["citation_support"] == 0.0

def test_polarity_two_way_and_irrelevant_negation_unknown():
    positive = claim_support("Purin gây bệnh.", [{"text": "Purin không gây bệnh."}])
    negative = claim_support("Purin không gây bệnh.", [{"text": "Purin gây bệnh."}])
    irrelevant = claim_support("Purin gây bệnh.", [{"text": "Không có dữ liệu về thời tiết."}])
    assert positive["contradicted_claims"] == 1
    assert negative["contradicted_claims"] == 1
    assert irrelevant["unknown_claims"] == 1


def test_rest_evaluator_fail_open_and_caps(monkeypatch):
    import src.backend.pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module, "QdrantClient", lambda path: object())
    import src.backend.app as app_module
    seen = {}
    def raising(question, answer, contexts, sufficient):
        seen["sizes"] = (len(question), len(answer), len(contexts))
        raise RuntimeError("test")
    monkeypatch.setattr(app_module, "evaluate_response", raising)
    result = app_module._evaluate("q" * 5000, "a" * 20000, [{"text": "x"}] * 20, True)
    assert result["status"] == "unavailable"
    assert seen["sizes"] == (4000, 12000, 10)

def test_public_support_has_no_raw_claim_text():
    result = evaluate_response("purin", "Purin hòa vào nước.", contexts(), True)
    assert "claim" not in result["claim_support"]["claims"][0]


def test_claim_parser_keeps_newline_and_bullets():
    result = claim_support("- Purin hòa vào nước khi hầm lâu\n- Uống đủ nước hỗ trợ cơ thể", [{"text": "Purin hòa vào nước khi hầm lâu."}, {"text": "Uống đủ nước hỗ trợ cơ thể."}])
    assert result["claim_count"] == 2
    assert result["supported_claims"] == 2

def test_prefix_postfix_middle_and_orphan_citations():
    contexts_ = [{"text": "Purin hòa vào nước."}]
    assert claim_support("Purin hòa vào nước.\n[Nguồn 1]", contexts_)["claims"][0]["citation_support"]
    assert claim_support("[Nguồn 1] Purin hòa vào nước.", contexts_)["claims"][0]["citation_support"]
    assert claim_support("Purin [Nguồn 1] hòa vào nước.", contexts_)["claims"][0]["citation_support"]
    orphan = claim_support("[Nguồn 1]", contexts_)
    assert orphan["claim_count"] == 1 and orphan["claims"][0]["citation_support"] is False

def test_multi_source_conflicting_and_irrelevant_negation_window():
    conflicting = claim_support("Purin gây ziekte.", [{"text": "Purin gây ziekte."}, {"text": "Purin không gây ziekte. Nước uống không lạnh."}])
    assert conflicting["conflicting_claims"] == 1
    irrelevant = claim_support("Purin gây ziekte.", [{"text": "Purin gây ziekte. Thời tiết không lạnh."}])
    assert irrelevant["contradicted_claims"] == 0 and irrelevant["supported_claims"] == 1


def test_fastapi_rest_additive_schema_with_mock_pipeline(monkeypatch):
    import src.backend.pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module, "QdrantClient", lambda path: object())
    import src.backend.app as app_module
    class FakePipeline:
        def search_with_evidence(self, question, top_k):
            return ([{"article_id": "article:a", "rank": 1, "citation_rank": 1, "text": "Purin hòa vào nước."}], True, 3.0, 2.5)
        def generate(self, question, contexts):
            return "Purin hòa vào nước. [Nguồn 1]"
        def close(self): pass
    monkeypatch.setattr(app_module, "pipeline", FakePipeline())
    from fastapi.testclient import TestClient
    response = TestClient(app_module.app).post("/api/qa/ask", json={"question": "purin", "top_k": 1})
    assert response.status_code == 200
    body = response.json()
    assert {"answer", "contexts", "retrieval", "citations", "confidence", "evaluation", "answer_status"} <= body.keys()

def test_websocket_stream_token_protocol_and_telemetry(monkeypatch, caplog):
    caplog.set_level("INFO")
    import src.backend.app as app_module
    class FakePipeline:
        def search_with_evidence(self, question, top_k):
            return ([{"rank": 1, "citation_rank": 1, "article_id": "a", "text": "Purin."}], True, 3.0, 2.5)
        def generate(self, question, contexts):
            return "Purin. [Nguồn 1]"
        def close(self): pass
    monkeypatch.setattr(app_module, "pipeline", FakePipeline())
    from fastapi.testclient import TestClient
    with TestClient(app_module.app) as client:
        with client.websocket_connect("/api/qa/stream") as socket:
            socket.send_json({"question": "purin"})
            tokens = [socket.receive_text() for _ in range(3)]
    assert "".join(tokens) .strip() == "Purin. [Nguồn 1]"
    assert "stream evaluation" in caplog.text


def test_legacy_rag_handler_additive_schema_and_latency(monkeypatch):
    import io, json
    import src.backend.pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module, "QdrantClient", lambda path: object())
    import src.backend.legacy_http as legacy
    class FakePipeline:
        def search_with_evidence(self, question, top_k):
            return ([{"rank": 1, "citation_rank": 1, "article_id": "a", "text": "Purin."}], True, 3.0, 2.5)
            return ([{"rank": 1, "citation_rank": 1, "article_id": "a", "text": "Purin."}], True, 3.0, 2.5)
        def generate(self, question, contexts): return "Purin. [Nguồn 1]"
        def close(self): pass
    monkeypatch.setattr(legacy, "PIPELINE", FakePipeline())
    class Handler(legacy.RagHandler):
        def __init__(self):
            self.headers = {"Content-Length": "0"}
            self.path = "/ask"
            self.wfile = io.BytesIO()
        def send_response(self, status): self.status = status
        def send_header(self, *args): pass
        def end_headers(self): pass
    body = json.dumps({"question": "purin", "top_k": 1}).encode()
    handler = Handler(); handler.headers = {"Content-Length": str(len(body))}; handler.rfile = io.BytesIO(body)
    handler.do_POST()
    payload = json.loads(handler.wfile.getvalue())
    assert handler.status == 200
    assert payload is not None and payload["confidence_deprecated"] is True and "evaluation_latency_ms" in payload["evaluation"]

def test_same_source_opposite_sentence_order_both_conflicting():
    claim = "Purin gây bệnh."
    first = claim_support(claim, [{"text": "Purin gây bệnh. Purin không gây bệnh."}])
    second = claim_support(claim, [{"text": "Purin không gây bệnh. Purin gây bệnh."}])
    assert first["conflicting_claims"] == 1
    assert second["conflicting_claims"] == 1

def test_multiline_context_first_and_last_supported():
    result = claim_support("Purin hòa vào nước. Uống đủ nước.", [{"text": "Purin hòa vào nước\nUống đủ nước"}])
    assert result["supported_claims"] == 2

def test_ten_bullets_supported_cited_no_abstention():
    contexts_ = [{"text": f"Thông tin mục {i}."} for i in range(1, 11)]
    answer = "\n".join(f"- Thông tin mục {i}. [Nguồn {i}]" for i in range(1, 11))
    result = evaluate_response("thông tin", answer, contexts_, True)
    assert result["claim_support"]["claim_count"] == 10
    assert result["claim_support"]["citation_index_validity"] == 1.0
    assert result["claim_support"]["citation_support"] == 1.0
    assert result["abstention_recommended"] is False


def test_fastapi_endpoint_evaluator_fail_open_200(monkeypatch):
    import src.backend.app as app_module
    class FakePipeline:
        def search_with_evidence(self, question, top_k):
            return ([{"rank": 1, "citation_rank": 1, "article_id": "a", "text": "Purin."}], False, 0.0, 0.0)
        def close(self): pass
    monkeypatch.setattr(app_module, "pipeline", FakePipeline())
    monkeypatch.setattr(app_module, "evaluate_response", lambda *args: (_ for _ in ()).throw(RuntimeError("boom")))
    from fastapi.testclient import TestClient
    response = TestClient(app_module.app).post("/api/qa/ask", json={"question": "unknown", "top_k": 1})
    assert response.status_code == 200
    assert response.json()["evaluation"]["status"] == "unavailable"
    assert response.json()["answer_status"] == "abstained"

def test_hf_raw_success_and_chat_template():
    from src.backend.pipeline import NewsPipeline
    pipeline = NewsPipeline.__new__(NewsPipeline)
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            self.called = (messages, kwargs)
            return "CHAT"
    class Generator:
        tokenizer = Tokenizer()
        def __call__(self, prompt, **kwargs):
            assert prompt == "CHAT"
            assert kwargs["add_special_tokens"] is False
            return [{"generated_text": "Aya answer"}]
    generator = Generator()
    pipeline._load_hf_generator = lambda: generator
    assert pipeline._generate_with_hf("question") == "Aya answer"
    assert generator.tokenizer.called[1]["add_generation_prompt"] is True


def test_hf_empty_retries_then_succeeds():
    from src.backend.pipeline import NewsPipeline
    pipeline = NewsPipeline.__new__(NewsPipeline)
    class Generator:
        tokenizer = None
        calls = 0
        def __call__(self, prompt, **kwargs):
            self.calls += 1
            return [{"generated_text": "" if self.calls == 1 else "retry answer"}]
    generator = Generator(); pipeline._load_hf_generator = lambda: generator
    assert pipeline._generate_with_hf("question") == "retry answer"
    assert generator.calls == 2


def test_hf_persistent_empty_is_stable_unavailable_error():
    import pytest
    from src.backend.pipeline import LLMUnavailableError, NewsPipeline
    pipeline = NewsPipeline.__new__(NewsPipeline)
    class Generator:
        tokenizer = None
        def __call__(self, prompt, **kwargs): return [{"generated_text": ""}]
    pipeline._load_hf_generator = lambda: Generator()
    with pytest.raises(LLMUnavailableError, match="empty answer after retry"):
        pipeline._generate_with_hf("question")

def test_fastapi_generated_path_evaluator_fail_open(monkeypatch):
    import src.backend.app as app_module
    class FakePipeline:
        def search_with_evidence(self, question, top_k):
            return ([{"rank": 1, "citation_rank": 1, "article_id": "a", "text": "Purin."}], True, 3.0, 2.5)
        def generate(self, question, contexts): return "Purin."
        def close(self): pass
    monkeypatch.setattr(app_module, "pipeline", FakePipeline())
    monkeypatch.setattr(app_module, "evaluate_response", lambda *args: (_ for _ in ()).throw(RuntimeError("boom")))
    from fastapi.testclient import TestClient
    response = TestClient(app_module.app).post("/api/qa/ask", json={"question": "purin", "top_k": 1})
    assert response.status_code == 200
    assert response.json()["answer_status"] == "generated"
    assert response.json()["evaluation"]["status"] == "unavailable"

def test_ten_numbered_items_keep_all_claims():
    contexts_ = [{"text": f"Thông tin mục {i}."} for i in range(1, 11)]
    answer = "\n".join(f"{i}. Thông tin mục {i}. [Nguồn {i}]" for i in range(1, 11))
    result = evaluate_response("thông tin", answer, contexts_, True)
    assert result["claim_support"]["claim_count"] == 10
    assert result["claim_support"]["citation_support"] == 1.0

def test_fastapi_generation_unavailable_controlled_200(monkeypatch):
    import src.backend.app as app_module
    from src.backend.pipeline import LLMUnavailableError
    class FakePipeline:
        def search_with_evidence(self, question, top_k):
            return ([{"rank": 1, "citation_rank": 1, "article_id": "a", "text": "Purin."}], True, 3.0, 2.5)
        def generate(self, question, contexts): raise LLMUnavailableError("stable")
        def close(self): pass
    monkeypatch.setattr(app_module, "pipeline", FakePipeline())
    from fastapi.testclient import TestClient
    response = TestClient(app_module.app).post("/api/qa/ask", json={"question": "purin", "top_k": 1})
    assert response.status_code == 200
    assert response.json()["answer_status"] == "generation_unavailable"
    assert response.json()["evaluation"]["status"] == "skipped"
    assert "stable" not in response.text

def test_hf_template_runtime_error_falls_back_plain_prompt():
    from src.backend.pipeline import NewsPipeline
    pipeline = NewsPipeline.__new__(NewsPipeline)
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs): raise RuntimeError("jinja TemplateError")
    assert pipeline._hf_input("prompt", Tokenizer()) == ("prompt", False)


def test_hf_runtime_and_oom_fail_stable_unavailable():
    import pytest
    from src.backend.pipeline import LLMUnavailableError, NewsPipeline
    for error in (RuntimeError("generation failed"), RuntimeError("CUDA out of memory")):
        pipeline = NewsPipeline.__new__(NewsPipeline)
        class Generator:
            tokenizer = None
            def __call__(self, prompt, **kwargs): raise error
        pipeline._load_hf_generator = lambda: Generator()
        with pytest.raises(LLMUnavailableError, match="Hugging Face generation failed"):
            pipeline._generate_with_hf("prompt")


def test_nested_chat_extractor_uses_last_assistant_only():
    from src.backend.pipeline import NewsPipeline
    output = [{"generated_text": [{"role": "user", "content": "secret prompt"}, {"role": "assistant", "content": "first"}, {"role": "assistant", "content": "last"}]}]
    assert NewsPipeline._extract_generated_text(output) == "last"

def test_api_empty_content_stable_unavailable(monkeypatch):
    import sys, types, pytest
    from src.backend.pipeline import LLMUnavailableError, NewsPipeline
    import src.backend.pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module.config, "LLM_API_KEY", "test-key")
    class Response:
        ok = True
        def json(self): return {"choices": [{"message": {"content": ""}}]}
    fake_requests = types.SimpleNamespace(post=lambda *args, **kwargs: Response(), Timeout=TimeoutError, RequestException=Exception)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    pipeline = NewsPipeline.__new__(NewsPipeline)
    with pytest.raises(LLMUnavailableError, match="LLM API returned an empty answer"):
        pipeline._generate_with_api("prompt")

def test_generator_lazy_load_double_checked_lock_once():
    from concurrent.futures import ThreadPoolExecutor
    from src.backend.pipeline import NewsPipeline
    pipeline = NewsPipeline.__new__(NewsPipeline)
    import threading
    pipeline.generator = None
    pipeline._load_lock = threading.Lock()
    calls = []
    def impl():
        pipeline.generator = "generator"
        calls.append(1)
        return "generator"
    pipeline._load_hf_generator_impl = impl
    with ThreadPoolExecutor(max_workers=8) as executor:
        values = list(executor.map(lambda _: pipeline._load_hf_generator(), range(8)))
    assert values == ["generator"] * 8
    assert len(calls) == 1

def test_websocket_generation_unavailable_controlled_skipped_no_leak(monkeypatch, caplog):
    caplog.set_level("INFO")
    import src.backend.app as app_module
    from src.backend.pipeline import LLMUnavailableError
    class FakePipeline:
        def search_with_evidence(self, question, top_k):
            return ([{"rank": 1, "citation_rank": 1, "article_id": "a", "text": "Purin."}], True, 3.0, 2.5)
        def generate(self, question, contexts): raise LLMUnavailableError("secret raw")
        def close(self): pass
    monkeypatch.setattr(app_module, "pipeline", FakePipeline())
    from fastapi.testclient import TestClient
    with TestClient(app_module.app) as client:
        with client.websocket_connect("/api/qa/stream") as socket:
            socket.send_json({"question": "purin"})
            token = "".join(socket.receive_text() for _ in range(20))
    assert "Không thể tạo" in token
    assert "secret raw" not in caplog.text
    assert "stream evaluation" in caplog.text and "status=skipped" in caplog.text

def test_nested_assistant_content_blocks_concatenate_first_last():
    from src.backend.pipeline import NewsPipeline
    output = [{"generated_text": [{"role": "assistant", "content": [{"type": "text", "text": "first"}, {"type": "text", "text": "last"}]}]}]
    assert NewsPipeline._extract_generated_text(output) == "first\nlast"


def test_generate_prompt_and_provider_inputs_are_strings(monkeypatch):
    from src.backend.pipeline import NewsPipeline
    pipeline = NewsPipeline.__new__(NewsPipeline)
    contexts_ = [{"rank": 1, "citation_rank": 1, "text": "evidence", "article_id": "a"}]
    prompt = pipeline._build_generation_prompt("question", contexts_)
    assert isinstance(prompt, str)
    seen = []
    pipeline.generator_provider = "api"
    pipeline._generate_with_api = lambda value: (seen.append(value) or "answer")
    pipeline.generate("question", contexts_)
    assert isinstance(seen[-1], str)
    pipeline.generator_provider = "hf_model"
    pipeline._generate_with_hf = lambda value: (seen.append(value) or "answer")
    pipeline.generate("question", contexts_)
    assert isinstance(seen[-1], str)


def test_encoder_and_reranker_lazy_load_double_checked_lock_once():
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from src.backend.pipeline import NewsPipeline
    for attr, impl_name in (("encoder", "_load_encoder_impl"), ("reranker", "_load_reranker_impl")):
        pipeline = NewsPipeline.__new__(NewsPipeline)
        setattr(pipeline, attr, None); pipeline._load_lock = threading.Lock(); calls = []
        def impl(attr=attr):
            calls.append(1); setattr(pipeline, attr, attr); return attr
        setattr(pipeline, impl_name, impl)
        loader = getattr(pipeline, "_load_" + attr)
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda _: loader(), range(4)))
        assert len(calls) == 1

def test_legacy_generation_unavailable_controlled_skipped_no_leak(monkeypatch):
    import io, json
    import src.backend.pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module, "QdrantClient", lambda path: object())
    import src.backend.legacy_http as legacy
    from src.backend.pipeline import LLMUnavailableError
    class FakePipeline:
        def search_with_evidence(self, question, top_k):
            return ([{"rank": 1, "citation_rank": 1, "article_id": "a", "text": "Purin."}], True, 3.0, 2.5)
        def generate(self, question, contexts): raise LLMUnavailableError("secret legacy")
    monkeypatch.setattr(legacy, "PIPELINE", FakePipeline())
    class Handler(legacy.RagHandler):
        def __init__(self, body):
            self.headers = {"Content-Length": str(len(body))}; self.path = "/ask"; self.rfile = io.BytesIO(body); self.wfile = io.BytesIO()
        def send_response(self, status): self.status = status
        def send_header(self, *args): pass
        def end_headers(self): pass
    handler = Handler(json.dumps({"question": "purin", "top_k": 1}).encode()); handler.do_POST()
    payload = json.loads(handler.wfile.getvalue())
    assert handler.status == 200 and payload["answer_status"] == "generation_unavailable"
    assert payload["evaluation"]["status"] == "skipped" and payload["evaluation"]["abstention_recommended"] is True
    assert "secret legacy" not in handler.wfile.getvalue().decode()
