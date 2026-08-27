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
    args = parser.parse_args()

    os.environ.setdefault("CORS_ORIGINS", "*")
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
