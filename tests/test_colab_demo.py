import json
from io import BytesIO
import zipfile

from deployment.export_colab_data import create_bundle
from deployment.run_colab_demo import extract_tunnel_url, wait_for_health
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
