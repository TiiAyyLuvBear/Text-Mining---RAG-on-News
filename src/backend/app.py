from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import config
from .pipeline import LLMUnavailableError, NewsPipeline
from .evaluation import EVALUATION_VERSION, evaluate_response

logging.basicConfig(
    level=os.getenv("RAG_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

pipeline = NewsPipeline()
LOGGER = logging.getLogger("rag-api")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        yield
    finally:
        pipeline.close()


app = FastAPI(title="Vietnamese News QA API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.CORS_ORIGINS),
    allow_credentials="*" not in config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def log_configuration() -> None:
    LOGGER.info(
        "config | qdrant_path=%s | collection=%s | qdrant_exists=%s",
        config.QDRANT_PATH, config.COLLECTION, config.QDRANT_PATH.exists(),
    )
    LOGGER.info(
        "config | embedding_model=%s | embedding_device=%s | reranker_model=%s | reranker_device=%s | model_dtype=%s | rerank_batch=%d | rerank_max_length=%d | min_score=%.2f | min_margin=%.2f",
        config.EMBEDDING_MODEL, config.EMBEDDING_DEVICE, config.RERANKER_MODEL,
        config.RERANKER_DEVICE, config.MODEL_DTYPE, config.RERANK_BATCH_SIZE,
        config.RERANK_MAX_LENGTH, config.RERANK_MIN_SCORE, config.RERANK_MIN_MARGIN,
    )
    resolved_provider = config.resolve_llm_provider()
    if resolved_provider == "hf_model":
        LOGGER.info(
            "config | llm_provider=%s | hf_model=%s | hf_token_configured=%s | hf_device=%s | load_in_4bit=%s | max_new_tokens=%d | retrieval_top_k=%d | context_top_k=%d",
            resolved_provider, config.HF_LLM_MODEL, bool(config.HF_TOKEN),
            config.HF_LLM_DEVICE, config.HF_LLM_LOAD_IN_4BIT, config.HF_LLM_MAX_NEW_TOKENS,
            config.TOP_K_RETRIEVAL, config.TOP_K_CONTEXT,
        )
    else:
        LOGGER.info(
            "config | llm_provider=%s | generator_model=%s | llm_endpoint=%s | max_tokens=%d | timeout_s=%.1f | retrieval_top_k=%d | context_top_k=%d",
            resolved_provider, config.GENERATOR_MODEL, config.LLM_API_URL,
            config.LLM_MAX_TOKENS, config.LLM_TIMEOUT, config.TOP_K_RETRIEVAL, config.TOP_K_CONTEXT,
        )


log_configuration()


@app.get("/api/health")
def health():
    return {"status": "ok", "index_ready": pipeline.is_ready()}


@app.get("/documents")
def documents():
    return {"documents": []}



def _evaluate(question: str, answer: str, contexts: list[dict], sufficient: bool) -> dict:
    started = time.perf_counter()
    try:
        result = evaluate_response(question[:4000], answer[:12000], contexts[:10], sufficient)
        result["evaluation_latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return result
    except Exception:
        LOGGER.exception("evaluation failed; request remains available")
        return {"status": "unavailable", "evaluation_version": EVALUATION_VERSION, "abstention_recommended": not sufficient, "evaluation_latency_ms": round((time.perf_counter() - started) * 1000, 3)}

@app.post("/api/qa/ask")
@app.post("/ask")
def ask(request: AskRequest):
    started = time.perf_counter()
    LOGGER.info("request start | question_chars=%d | top_k=%d", len(request.question), request.top_k)
    contexts, sufficient, top_score, margin = pipeline.search_with_evidence(request.question, request.top_k)
    if sufficient:
        try:
            answer = pipeline.generate(request.question, contexts)
            answer_status = "generated"
        except LLMUnavailableError:
            answer = "Không thể tạo câu trả lời từ mô hình lúc này; dữ liệu vẫn đủ bằng chứng nhưng hệ thống không nhận được đầu ra hợp lệ."
            answer_status = "generation_unavailable"
            LOGGER.error("generation unavailable; returning controlled response")
    else:
        answer = "Không đủ thông tin trong dữ liệu được cung cấp để trả lời câu hỏi này một cách đáng tin cậy."
        answer_status = "abstained"
        LOGGER.warning(
            "abstention | top_score=%.4f | margin=%.4f | min_score=%.2f | min_margin=%.2f",
            top_score, margin, config.RERANK_MIN_SCORE, config.RERANK_MIN_MARGIN,
        )
    evaluation = ({"status": "skipped", "reason": "generation_unavailable", "evaluation_version": EVALUATION_VERSION, "abstention_recommended": True, "evaluation_latency_ms": 0.0}
                  if answer_status == "generation_unavailable" else _evaluate(request.question, answer, contexts, sufficient))
    # Legacy confidence retained as BGE gate; not calibrated answer confidence.
    confidence = 1.0 if sufficient else 0.0
    payload = {
        "answer": answer,
        "citations": [
            {
                "article_id": item.get("article_id"),
                "title": item.get("title"),
                "url": item.get("url"),
                "score": item.get("rerank_score"),
            }
            for item in contexts
        ],
        "retrieval": contexts,
        "contexts": contexts,
        "confidence": confidence,
        "confidence_percent": round(confidence * 100, 1),
        "confidence_deprecated": True,
        "confidence_method": "LEGACY: BGE evidence gate only; not answer factuality.",
        "evaluation": evaluation,
        "evidence_sufficient": sufficient,
        "evidence_status": getattr(pipeline, "last_evidence_quality", {}).get("status", "sufficient" if sufficient else "insufficient"),
        "evidence_quality": getattr(pipeline, "last_evidence_quality", {}),
        "rerank_top_score": top_score,
        "rerank_margin": margin,
        "rerank_margin_deprecated": True,
        "rerank_min_score": config.RERANK_MIN_SCORE,
        "rerank_min_margin": config.RERANK_MIN_MARGIN,
        "answer_status": answer_status,
        "response_time_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    LOGGER.info("request done | contexts=%d | response_ms=%.1f", len(contexts), payload["response_time_ms"])
    return payload


async def stream_answer(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            body = json.loads(await websocket.receive_text())
            question = str(body.get("text") or body.get("question") or "").strip()
            if not question:
                continue
            contexts, sufficient, top_score, margin = await asyncio.to_thread(pipeline.search_with_evidence, question, 5)
            if sufficient:
                try:
                    answer = await asyncio.to_thread(pipeline.generate, question, contexts)
                    answer_status = "generated"
                except LLMUnavailableError:
                    answer = "Không thể tạo câu trả lời từ mô hình lúc này; hệ thống không nhận được đầu ra hợp lệ."
                    answer_status = "generation_unavailable"
                    LOGGER.error("stream generation unavailable; returning controlled response")
            else:
                answer = "Không đủ thông tin trong dữ liệu được cung cấp để trả lời câu hỏi này một cách đáng tin cậy."
                answer_status = "abstained"
                LOGGER.warning("stream abstention | top_score=%.4f | margin=%.4f", top_score, margin)
            evaluation = ({"status": "skipped", "reason": "generation_unavailable", "evaluation_version": EVALUATION_VERSION, "abstention_recommended": True, "evaluation_latency_ms": 0.0}
                          if answer_status == "generation_unavailable" else await asyncio.to_thread(_evaluate, question, answer, contexts, sufficient))
            LOGGER.info("stream evaluation | status=%s | abstention=%s | support_coverage=%s | citation_support=%s | evaluation_ms=%s", evaluation.get("status", "ok"), evaluation.get("abstention_recommended"), (evaluation.get("claim_support") or {}).get("lexical_support_coverage"), (evaluation.get("claim_support") or {}).get("citation_support"), evaluation.get("evaluation_latency_ms"))
            for token in answer.split(" "):
                await websocket.send_text(token + " ")
                await asyncio.sleep(0.005)
    except WebSocketDisconnect:
        return


@app.websocket("/api/qa/stream")
async def qa_stream(websocket: WebSocket):
    await stream_answer(websocket)


@app.websocket("/chat/stream")
async def legacy_chat_stream(websocket: WebSocket):
    await stream_answer(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)


