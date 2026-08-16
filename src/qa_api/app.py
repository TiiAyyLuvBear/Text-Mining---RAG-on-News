from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .pipeline import IndexUnavailableError, LLMUnavailableError, NewsPipeline, RerankerError

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

FRONTEND_DIST = config.ROOT / "src" / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"
if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")
else:
    LOGGER.warning("React build not found at src/frontend/dist; API will run without the web UI")


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
    index_ready = pipeline.is_ready()
    return {"status": "ok" if index_ready else "degraded", "index_ready": index_ready}


@app.get("/documents")
def documents():
    return {"documents": []}


def _citations(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "article_id": item.get("article_id"),
            "title": item.get("title"),
            "url": item.get("url"),
            "score": item.get("rerank_score"),
        }
        for item in contexts
    ]


def _error_response(
    *,
    code: str,
    message: str,
    status_code: int,
    started: float,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "answer": "",
            "contexts": [],
            "citations": [],
            "confidence": 0.0,
            "evidence_sufficient": False,
            "answer_status": "error",
            "error": {"code": code, "message": message},
            "response_time_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    )


def process_question(request: AskRequest) -> dict[str, Any] | JSONResponse:
    started = time.perf_counter()
    question = request.question.strip()
    LOGGER.info("request start | question_chars=%d | top_k=%d", len(question), request.top_k)
    if not question:
        return _error_response(
            code="INVALID_QUESTION",
            message="Câu hỏi không được để trống.",
            status_code=400,
            started=started,
        )

    try:
        contexts, sufficient, top_score, margin = pipeline.search_with_evidence(question, request.top_k)
        if sufficient:
            answer = pipeline.generate(question, contexts)
            answer_status = "generated"
        else:
            answer = "Chưa đủ bằng chứng trong kho tin để trả lời câu hỏi này một cách đáng tin cậy."
            answer_status = "abstained"
            LOGGER.warning(
                "abstention | top_score=%.4f | margin=%.4f | min_score=%.2f | min_margin=%.2f",
                top_score, margin, config.RERANK_MIN_SCORE, config.RERANK_MIN_MARGIN,
            )
    except IndexUnavailableError:
        LOGGER.exception("Qdrant/index unavailable")
        return _error_response(
            code="INDEX_UNAVAILABLE",
            message="Kho tìm kiếm tin tức chưa sẵn sàng.",
            status_code=503,
            started=started,
        )
    except RerankerError:
        LOGGER.exception("Reranker unavailable")
        return _error_response(
            code="RERANKER_UNAVAILABLE",
            message="Không thể xếp hạng bằng chứng lúc này.",
            status_code=503,
            started=started,
        )
    except LLMUnavailableError:
        LOGGER.exception("LLM unavailable")
        return _error_response(
            code="LLM_UNAVAILABLE",
            message="Không thể tạo câu trả lời lúc này.",
            status_code=502,
            started=started,
        )
    except Exception:
        LOGGER.exception("Unexpected QA pipeline error")
        return _error_response(
            code="PIPELINE_ERROR",
            message="Hệ thống không thể xử lý câu hỏi lúc này.",
            status_code=500,
            started=started,
        )

    confidence = 1.0 if sufficient else 0.0
    payload = {
        "answer": answer,
        "contexts": contexts,
        "citations": _citations(contexts),
        # Kept temporarily for clients that still consume the old field name.
        "retrieval": contexts,
        "confidence": confidence,
        "confidence_method": "binary_evidence_gate",
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


@app.post("/api/qa/ask")
@app.post("/ask", include_in_schema=False)
def ask(request: AskRequest):
    return process_question(request)


async def stream_answer(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            body = json.loads(await websocket.receive_text())
            question = str(body.get("text") or body.get("question") or "").strip()
            if not question:
                continue
            contexts, sufficient, top_score, margin = await asyncio.to_thread(
                pipeline.search_with_evidence, question, 5
            )
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


@app.get("/{frontend_path:path}", include_in_schema=False)
def frontend(frontend_path: str):
    # API typos must remain JSON 404s instead of falling through to the SPA.
    if frontend_path == "api" or frontend_path.startswith("api/"):
        return JSONResponse(status_code=404, content={"detail": "API endpoint not found"})
    if not FRONTEND_INDEX.is_file():
        return JSONResponse(
            status_code=404,
            content={"detail": "React frontend is not built. Run npm run build in src/frontend."},
        )
    candidate = FRONTEND_DIST / frontend_path
    if frontend_path and candidate.is_file() and candidate.resolve().is_relative_to(FRONTEND_DIST.resolve()):
        return FileResponse(candidate)
    return FileResponse(FRONTEND_INDEX)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
