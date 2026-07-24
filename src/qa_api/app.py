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
from .pipeline import NewsPipeline

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
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def log_configuration() -> None:
    LOGGER.info(
        "config | qdrant_path=%s | collection=%s | qdrant_exists=%s",
        config.QDRANT_PATH, config.COLLECTION, config.QDRANT_PATH.exists(),
    )
    LOGGER.info(
        "config | embedding_model=%s | reranker_model=%s | rerank_batch=%d | rerank_max_length=%d | min_score=%.2f | min_margin=%.2f",
        config.EMBEDDING_MODEL, config.RERANKER_MODEL,
        config.RERANK_BATCH_SIZE, config.RERANK_MAX_LENGTH, config.RERANK_MIN_SCORE, config.RERANK_MIN_MARGIN,
    )
    LOGGER.info(
        "config | generator_model=%s | llm_endpoint=%s | max_tokens=%d | timeout_s=%.1f | retrieval_top_k=%d | context_top_k=%d",
        config.GENERATOR_MODEL, config.LLM_API_URL, config.LLM_MAX_TOKENS,
        config.LLM_TIMEOUT, config.TOP_K_RETRIEVAL, config.TOP_K_CONTEXT,
    )


log_configuration()


@app.get("/api/health")
def health():
    return {"status": "ok", "index_ready": pipeline.is_ready()}


@app.get("/documents")
def documents():
    return {"documents": []}


@app.post("/api/qa/ask")
def ask(request: AskRequest):
    started = time.perf_counter()
    LOGGER.info("request start | question_chars=%d | top_k=%d", len(request.question), request.top_k)
    contexts = pipeline.search(request.question, request.top_k)
    sufficient, top_score, margin = pipeline.evidence_quality(contexts)
    if sufficient:
        answer = pipeline.generate(request.question, contexts)
        answer_status = "generated"
    else:
        answer = "Không đủ thông tin trong dữ liệu được cung cấp để trả lời câu hỏi này một cách đáng tin cậy."
        answer_status = "abstained"
        LOGGER.warning(
            "abstention | top_score=%.4f | margin=%.4f | min_score=%.2f | min_margin=%.2f",
            top_score, margin, config.RERANK_MIN_SCORE, config.RERANK_MIN_MARGIN,
        )
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
        "confidence": confidence,
        "confidence_percent": round(confidence * 100, 1),
        "confidence_method": "BGE gate: top_score >= ngưỡng và top_score - second_score >= margin",
        "evidence_sufficient": sufficient,
        "rerank_top_score": top_score,
        "rerank_margin": margin,
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
            contexts = await asyncio.to_thread(pipeline.search, question, 5)
            sufficient, top_score, margin = pipeline.evidence_quality(contexts)
            if sufficient:
                answer = await asyncio.to_thread(pipeline.generate, question, contexts)
            else:
                answer = "Không đủ thông tin trong dữ liệu được cung cấp để trả lời câu hỏi này một cách đáng tin cậy."
                LOGGER.warning("stream abstention | top_score=%.4f | margin=%.4f", top_score, margin)
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
