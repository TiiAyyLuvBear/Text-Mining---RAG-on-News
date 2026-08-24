"""Canonical source identity shared by serving and telemetry."""
from __future__ import annotations
import hashlib
from typing import Any


def source_key(item: dict[str, Any], index: int = 0) -> str:
    for field in ("article_id", "url", "chunk_id"):
        value = str(item.get(field) or "").strip()
        if value:
            return f"{field}:{value}"
    digest = hashlib.sha1(str(item.get("text", "")).encode("utf-8", "replace")).hexdigest()[:12]
    return f"anonymous:{index}:{digest}"
