"""Per-request JSON traces for inspecting the RAG pipeline."""

from __future__ import annotations

import contextvars
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from . import config


LOGGER = logging.getLogger("rag-api.request-trace")
_ACTIVE_TRACE: contextvars.ContextVar[RequestTrace | None] = contextvars.ContextVar(
    "active_request_trace", default=None,
)


def _json_safe(value: Any) -> Any:
    """Keep trace persistence best-effort even for model-library objects."""
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        return str(value)


class RequestTrace:
    """Owns one trace directory and ordered JSON phase outputs."""

    def __init__(self, *, endpoint: str, question: str, top_k: int) -> None:
        created_at = datetime.now(timezone.utc)
        self.request_id = f"{created_at:%Y%m%dT%H%M%S_%fZ}_{uuid4().hex[:8]}"
        self.path = config.REQUEST_TRACE_DIR / self.request_id
        self.path.mkdir(parents=True, exist_ok=False)
        self._phase_number = 0
        self.write(
            "request",
            {
                "request_id": self.request_id,
                "created_at": created_at.isoformat(),
                "endpoint": endpoint,
                "question": question,
                "top_k": top_k,
            },
        )

    def write(self, phase: str, payload: Any) -> Path:
        self._phase_number += 1
        destination = self.path / f"{self._phase_number:02d}_{phase}.json"
        temporary = destination.with_suffix(".json.tmp")
        try:
            temporary.write_text(
                json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(destination)
        except OSError:
            LOGGER.exception("request trace write failed | phase=%s", phase)
        return destination


def start_trace(*, endpoint: str, question: str, top_k: int) -> tuple[RequestTrace | None, contextvars.Token]:
    """Create and activate a trace; tracing must never break a user request."""
    if not config.REQUEST_TRACE_ENABLED:
        return None, _ACTIVE_TRACE.set(None)
    try:
        trace = RequestTrace(endpoint=endpoint, question=question, top_k=top_k)
    except OSError:
        LOGGER.exception("request trace start failed")
        return None, _ACTIVE_TRACE.set(None)
    return trace, _ACTIVE_TRACE.set(trace)


def current_trace() -> RequestTrace | None:
    return _ACTIVE_TRACE.get()


def trace_phase(phase: str, payload: Any) -> None:
    trace = current_trace()
    if trace is not None:
        trace.write(phase, payload)


def finish_trace(trace: RequestTrace | None, token: contextvars.Token, *, status: str, payload: Any) -> None:
    try:
        if trace is not None:
            trace.write("response", {"request_id": trace.request_id, "status": status, "payload": payload})
    finally:
        _ACTIVE_TRACE.reset(token)
