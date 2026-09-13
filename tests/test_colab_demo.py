import json
from io import BytesIO
import os
import subprocess
import sys
import zipfile

import pytest

from deployment.export_colab_data import create_bundle
from deployment.run_colab_demo import (
    configure_colab_environment,
    extract_tunnel_url,
    wait_for_health,
)
from src.backend.config import parse_cors_origins


def test_parse_cors_origins_supports_public_demo_and_lists():
    assert parse_cors_origins("*") == ("*",)
    assert parse_cors_origins("https://ui.example/, http://localhost:5173") == (
        "https://ui.example",
        "http://localhost:5173",
    )


def test_extract_tunnel_url_ignores_other_log_text():
    assert extract_tunnel_url("INF https://quiet-tree.trycloudflare.com ready") == (
        "https://quiet-tree.trycloudflare.com"
    )
    assert extract_tunnel_url("connection registered") is None


def test_wait_for_health_reads_json(monkeypatch):
    payload = BytesIO(b'{"status":"ok","index_ready":true}')
    monkeypatch.setattr(
        "deployment.run_colab_demo.urllib.request.urlopen", lambda *args, **kwargs: payload
    )
    assert wait_for_health("http://127.0.0.1:8000/api/health", timeout=1)["index_ready"]


def test_colab_environment_forces_hf_before_backend_child_import(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "api")
    monkeypatch.setenv("HF_LLM_MODEL", "")

    configured = configure_colab_environment()

    assert configured["LLM_PROVIDER"] == "hf_model"
    assert configured["HF_LLM_MODEL"] == "Qwen/Qwen2.5-7B-Instruct"
    assert configured["HF_LLM_DEVICE"] == "cuda:0"
    assert configured["HF_LLM_LOAD_IN_4BIT"] == "true"
    assert configured["EMBEDDING_DEVICE"] == "cuda:0"
    assert configured["RERANKER_DEVICE"] == "cuda:0"


def _config_snapshot(**environment):
    env = os.environ.copy()
    env.update(environment)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from src.backend import config; "
            "print(json.dumps({'provider': config.resolve_llm_provider(), "
            "'hf_model': config.HF_LLM_MODEL, 'api_key': bool(config.LLM_API_KEY)}))",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(completed.stdout)


def test_explicit_hf_provider_uses_qwen_without_api_key():
    snapshot = _config_snapshot(
        LLM_PROVIDER="hf_model",
        HF_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct",
        LLM_API_KEY="",
        ANTHROPIC_API_KEY="",
    )
    assert snapshot == {
        "provider": "hf_model",
        "hf_model": "Qwen/Qwen2.5-7B-Instruct",
        "api_key": False,
    }


def test_explicit_api_and_auto_selection_remain_intact():
    assert _config_snapshot(LLM_PROVIDER="api", HF_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct")["provider"] == "api"
    assert _config_snapshot(LLM_PROVIDER="auto", HF_LLM_MODEL="")["provider"] == "api"
    assert _config_snapshot(LLM_PROVIDER="auto", HF_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct")["provider"] == "hf_model"


def test_hf_generation_failure_never_falls_back_to_api():
    from src.backend.pipeline import LLMUnavailableError, NewsPipeline

    pipeline = NewsPipeline.__new__(NewsPipeline)
    pipeline.generator_provider = "hf_model"
    pipeline._generate_with_hf = lambda _: (_ for _ in ()).throw(LLMUnavailableError("hf failed"))
    api_calls = []
    pipeline._generate_with_api = lambda _: api_calls.append(True)

    with pytest.raises(LLMUnavailableError, match="hf failed"):
        pipeline.generate("Câu hỏi", [{"citation_rank": 1, "text": "Tư liệu"}])
    assert api_calls == []


def test_create_bundle_normalizes_paths_and_excludes_lock(tmp_path):
    qdrant = tmp_path / "source-index"
    qdrant.mkdir()
    (qdrant / "meta.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (qdrant / ".lock").write_text("locked", encoding="utf-8")
    bm25 = tmp_path / "source.pkl"
    bm25.write_bytes(b"index")

    output = create_bundle(tmp_path / "bundle.zip", qdrant, bm25)

    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            "data/qdrant_news/meta.json",
            "data/qdrant_news_bm25.pkl",
        }
