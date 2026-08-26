"""HTTP backend kept for the Vite frontend's legacy ``POST /ask`` contract.

Retrieval and reranking run at request time through :mod:`src.backend.pipeline`.
"""

from __future__ import annotations

import json
import logging
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from src.backend import config
from src.backend.pipeline import LLMUnavailableError, NewsPipeline
from src.backend.evaluation import EVALUATION_VERSION, evaluate_response

DEFAULT_HOST = os.getenv("RAG_API_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("RAG_API_PORT", "8000"))
MAX_TOP_K = 10

logging.basicConfig(
    level=os.getenv("RAG_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("rag-backend")
PIPELINE = NewsPipeline()


class RagHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send_json(200, {"ok": True})

    def do_POST(self) -> None:
        started = time.perf_counter()
        if self.path.rstrip("/") not in {"/ask", "/api/qa/ask"}:
            self._send_json(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            question = str(payload.get("question", "")).strip()
            top_k = int(payload.get("top_k", config.TOP_K_CONTEXT))
            if not question:
                self._send_json(400, {"error": "Missing question"})
                return
            if not 1 <= top_k <= MAX_TOP_K:
                self._send_json(400, {"error": f"top_k must be between 1 and {MAX_TOP_K}"})
                return

            # Runtime path: E5 query embedding, Qdrant search, BGE reranking.
            contexts, sufficient, top_score, margin = PIPELINE.search_with_evidence(question, top_k)
            if sufficient:
                try:
                    answer = PIPELINE.generate(question, contexts)
                    answer_status = "generated"
                except LLMUnavailableError:
                    answer = "Không thể tạo câu trả lời từ mô hình lúc này; dữ liệu vẫn đủ bằng chứng nhưng hệ thống không nhận được đầu ra hợp lệ."
                    answer_status = "generation_unavailable"
                    LOGGER.error("generation unavailable; returning controlled response")
            else:
                answer = "Không đủ thông tin trong dữ liệu được cung cấp để trả lời câu hỏi này một cách đáng tin cậy."
                answer_status = "abstained"
                LOGGER.warning("abstention | top_score=%.4f | margin=%.4f", top_score, margin)

            evaluation_started = time.perf_counter()
            if answer_status == "generation_unavailable":
                evaluation = {"status": "skipped", "reason": "generation_unavailable", "evaluation_version": EVALUATION_VERSION, "abstention_recommended": True, "evaluation_latency_ms": 0.0}
            else:
                try:
                    evaluation = evaluate_response(question[:4000], answer[:12000], contexts[:10], sufficient)
                    evaluation["evaluation_latency_ms"] = round((time.perf_counter() - evaluation_started) * 1000, 3)
                except Exception:
                    LOGGER.exception("evaluation failed; request remains available")
                    evaluation = {"status": "unavailable", "evaluation_version": EVALUATION_VERSION, "abstention_recommended": not sufficient, "evaluation_latency_ms": round((time.perf_counter() - evaluation_started) * 1000, 3)}

            LOGGER.info("evaluation | status=%s | abstention=%s | support_coverage=%s | citation_support=%s | evaluation_ms=%s", evaluation.get("status", "ok"), evaluation.get("abstention_recommended"), (evaluation.get("claim_support") or {}).get("lexical_support_coverage"), (evaluation.get("claim_support") or {}).get("citation_support"), evaluation.get("evaluation_latency_ms"))
            elapsed_ms = (time.perf_counter() - started) * 1000
            self._send_json(200, {
                "answer": answer,
                "contexts": contexts,
                "citations": [
                    {
                        "article_id": item.get("article_id"),
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "score": item.get("rerank_score"),
                    }
                    for item in contexts
                ],
                "confidence": 1.0 if sufficient else 0.0,
                "confidence_percent": 100.0 if sufficient else 0.0,
                "confidence_deprecated": True,
                "confidence_method": "LEGACY: BGE evidence gate only; not answer factuality.",
                "evaluation": evaluation,
                "evidence_sufficient": sufficient,
                "evidence_status": getattr(PIPELINE, "last_evidence_quality", {}).get("status", "sufficient" if sufficient else "insufficient"),
                "evidence_quality": getattr(PIPELINE, "last_evidence_quality", {}),
                "rerank_top_score": top_score,
                "rerank_margin": margin,
                "rerank_margin_deprecated": True,
                "rerank_min_score": config.RERANK_MIN_SCORE,
                "rerank_min_margin": config.RERANK_MIN_MARGIN,
                "answer_status": answer_status,
                "response_time_ms": round(elapsed_ms, 1),
            })
            LOGGER.info("POST /ask | contexts=%d | response_ms=%.1f", len(contexts), elapsed_ms)
        except (TypeError, ValueError) as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception:
            LOGGER.exception("POST /ask failed")
            self._send_json(500, {"error": "Backend processing failed"})


def main() -> None:
    server = ThreadingHTTPServer((DEFAULT_HOST, DEFAULT_PORT), RagHandler)
    LOGGER.info("Backend listening on http://%s:%s", DEFAULT_HOST, DEFAULT_PORT)
    LOGGER.info(
        "config | index_ready=%s | collection=%s | retrieval_top_k=%d | rerank_model=%s",
        PIPELINE.is_ready(), config.COLLECTION, config.TOP_K_RETRIEVAL, config.RERANKER_MODEL,
    )
    try:
        server.serve_forever()
    finally:
        PIPELINE.close()
        server.server_close()


if __name__ == "__main__":
    main()


