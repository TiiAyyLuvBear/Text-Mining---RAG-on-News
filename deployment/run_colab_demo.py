"""Run FastAPI and expose it through a temporary Cloudflare Quick Tunnel."""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
TUNNEL_URL_PATTERN = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def configure_colab_environment(
    *,
    llm_provider: str = "hf_model",
    hf_llm_model: str = DEFAULT_HF_MODEL,
    hf_llm_device: str = "cuda:0",
    load_in_4bit: bool = True,
) -> dict[str, str]:
    """Set backend configuration before the Uvicorn child imports config.py."""
    values = {
        "QDRANT_PATH": "data/qdrant_news",
        "BM25_INDEX_PATH": "data/qdrant_news_bm25.pkl",
        "QDRANT_COLLECTION": "news_bge_token",
        "MODEL_DEVICE": "cuda:0",
        "MODEL_DTYPE": "float16",
        "EMBEDDING_DEVICE": "cuda:0",
        "RERANKER_DEVICE": "cuda:0",
        "LLM_PROVIDER": llm_provider,
        "HF_LLM_MODEL": hf_llm_model,
        "HF_LLM_DEVICE": hf_llm_device,
        "HF_LLM_LOAD_IN_4BIT": "true" if load_in_4bit else "false",
        "HF_LLM_4BIT_QUANT_TYPE": "nf4",
        "HF_LLM_4BIT_USE_DOUBLE_QUANT": "true",
        "HF_LLM_4BIT_COMPUTE_DTYPE": "float16",
        "HF_LLM_MAX_NEW_TOKENS": "700",
        "CORS_ORIGINS": "*",
    }
    os.environ.update(values)
    return values


def validate_colab_runtime(*, llm_provider: str, hf_llm_device: str, load_in_4bit: bool) -> None:
    """Fail before tunnel startup when explicitly requested CUDA cannot work."""
    if llm_provider != "hf_model" or not hf_llm_device.startswith("cuda"):
        return
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Colab Hugging Face generation.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a Colab GPU runtime before starting the demo.")
    if load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("bitsandbytes is required for 4-bit Hugging Face generation.") from exc
    print(f"GPU={torch.cuda.get_device_name(0)}", flush=True)


def extract_tunnel_url(line: str) -> str | None:
    match = TUNNEL_URL_PATTERN.search(line)
    return match.group(0) if match else None


def wait_for_health(url: str, timeout: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                import json

                return json.load(response)
        except Exception as exc:  # Service may still be loading.
            last_error = exc
            time.sleep(1)
    raise TimeoutError(f"API did not become healthy within {timeout:.0f}s: {last_error}")


def ensure_cloudflared(requested_path: str | None = None) -> str:
    if requested_path:
        path = Path(requested_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"cloudflared not found: {path}")
        return str(path)
    installed = shutil.which("cloudflared")
    if installed:
        return installed
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise RuntimeError("Automatic cloudflared download supports Linux x86_64 only.")
    path = Path(tempfile.gettempdir()) / "cloudflared"
    if not path.is_file():
        print("Downloading cloudflared...", flush=True)
        urllib.request.urlretrieve(CLOUDFLARED_URL, path)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cloudflared", default=None)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--llm-provider", choices=("hf_model", "api", "auto"), default="hf_model")
    parser.add_argument("--hf-llm-model", default=DEFAULT_HF_MODEL)
    parser.add_argument("--hf-llm-device", default="cuda:0")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    configured = configure_colab_environment(
        llm_provider=args.llm_provider,
        hf_llm_model=args.hf_llm_model,
        hf_llm_device=args.hf_llm_device,
        load_in_4bit=not args.no_4bit,
    )
    validate_colab_runtime(
        llm_provider=args.llm_provider,
        hf_llm_device=args.hf_llm_device,
        load_in_4bit=not args.no_4bit,
    )
    print(
        "BACKEND_GENERATOR_CONFIG="
        f"provider={configured['LLM_PROVIDER']} "
        f"model={configured['HF_LLM_MODEL']} "
        f"device={configured['HF_LLM_DEVICE']} "
        f"load_in_4bit={configured['HF_LLM_LOAD_IN_4BIT']}",
        flush=True,
    )
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.backend.app:app", "--host", "0.0.0.0", "--port", str(args.port)],
    )
    tunnel: subprocess.Popen[str] | None = None
    try:
        health = wait_for_health(
            f"http://127.0.0.1:{args.port}/api/health", timeout=args.startup_timeout
        )
        if not health.get("index_ready"):
            raise RuntimeError("RAG index is not ready. Import or build index before starting demo.")
        cloudflared = ensure_cloudflared(args.cloudflared)
        tunnel = subprocess.Popen(
            [cloudflared, "tunnel", "--url", f"http://127.0.0.1:{args.port}", "--no-autoupdate"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        announced = False
        assert tunnel.stdout is not None
        for line in tunnel.stdout:
            print(line, end="", flush=True)
            url = extract_tunnel_url(line)
            if url and not announced:
                print(f"\nPUBLIC_API_URL={url}", flush=True)
                print(f"FRONTEND_ENV=VITE_API_BASE_URL={url}\n", flush=True)
                announced = True
        return tunnel.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        if tunnel and tunnel.poll() is None:
            tunnel.terminate()
        if api.poll() is None:
            api.terminate()
            try:
                api.wait(timeout=10)
            except subprocess.TimeoutExpired:
                api.kill()


if __name__ == "__main__":
    raise SystemExit(main())
