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


def _adaptive_response(question: str, top_k: int) -> tuple[dict, list[dict]]:
    """Use the one canonical evidence-routed generation flow."""
    adaptive = getattr(pipeline, "search_adaptive", None)
    if callable(adaptive):
        decision = adaptive(question, top_k).model_dump()
        contexts = list(getattr(pipeline, "last_ranked_contexts", []))
        return decision, contexts
    # Compatibility for old test doubles and external callers implementing the
    # historic pipeline interface. Real NewsPipeline instances always use the
    # adaptive branch above.
    contexts, sufficient, _, _ = pipeline.search_with_evidence(question, top_k)
    try:
        answer = pipeline.generate(question, contexts) if sufficient else "Không đủ thông tin trong dữ liệu được cung cấp để trả lời câu hỏi này một cách đáng tin cậy."
        status = "generated" if sufficient else "abstained"
    except LLMUnavailableError:
        answer, status = "Không thể tạo câu trả lời từ mô hình lúc này; dữ liệu vẫn đủ bằng chứng nhưng hệ thống không nhận được đầu ra hợp lệ.", "generation_unavailable"
    decision = {"decision": "ANSWER" if status == "generated" else "REFUSE", "answer": answer, "citations": [], "refusal_reason": "" if status == "generated" else status, "missing_evidence": [], "_legacy_status": status}
    return decision, contexts


def _answer_status(decision: dict) -> str:
    if decision.get("_legacy_status"):
        return str(decision["_legacy_status"])
    if decision.get("decision") == "ANSWER":
        return "generated"
    if decision.get("failure_category") == "EVIDENCE":
        return "abstained"
    if decision.get("failure_category") == "GENERATOR":
        return "generation_unavailable"
    return "refused"


def _evidence_sufficient(decision: dict, route_decision: dict) -> bool:
    route = str((route_decision or {}).get("route") or "")
    if route:
        return route in {"SINGLE_DOC", "REQUIRES_MULTI_DOC"}
    return decision.get("decision") == "ANSWER" or decision.get("failure_category") in {
        "GENERATOR", "VERIFICATION",
    }


def _generation_contexts(retrieval: list[dict], decision: dict) -> list[dict]:
    route = decision.get("route_decision") or getattr(pipeline, "last_route_decision", {})
    selected = {str(value) for value in route.get("selected_article_ids", [])}
    planned = list(getattr(pipeline, "last_planned_contexts", []))
    if not selected:
        selected = {str(item.get("article_id") or "") for item in planned}
    grouped = [item for item in retrieval if str(item.get("article_id") or "") in selected]
    if grouped and any(item.get("evidence_chunks") for item in grouped):
        return grouped
    if planned:
        return planned
    return grouped

@app.post("/api/qa/ask")
@app.post("/ask")
def ask(request: AskRequest):
    started = time.perf_counter()
    LOGGER.info("request start | question_chars=%d | top_k=%d", len(request.question), request.top_k)
    decision, contexts = _adaptive_response(request.question, request.top_k)
    answer = decision["answer"]
    answer_status = _answer_status(decision)
    route_decision = getattr(pipeline, "last_route_decision", {})
    selected_contexts = _generation_contexts(contexts, {**decision, "route_decision": route_decision})
    evaluation = ({"status": "skipped", "reason": "generation_unavailable", "evaluation_version": EVALUATION_VERSION, "abstention_recommended": True, "evaluation_latency_ms": 0.0}
                  if answer_status == "generation_unavailable" else _evaluate(request.question, answer, selected_contexts, decision["decision"] == "ANSWER"))
    payload = {
        **{key: value for key, value in decision.items() if key != "_legacy_status"},
        "answer": answer,
        "retrieval": contexts,
        "contexts": selected_contexts,
        "evidence_plan": getattr(pipeline, "last_evidence_plan", {}),
        "coverage_matrix": getattr(pipeline, "last_coverage_matrix", []),
        "confidence": 1.0 if decision["decision"] == "ANSWER" else 0.0,
        "confidence_percent": 100.0 if decision["decision"] == "ANSWER" else 0.0,
        "confidence_deprecated": True,
        "confidence_method": "LEGACY: BGE evidence gate only; not answer factuality.",
        "evaluation": evaluation,
        "evidence_sufficient": _evidence_sufficient(decision, route_decision),
        "route_decision": route_decision,
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
            decision, contexts = await asyncio.to_thread(_adaptive_response, question, 5)
            answer = decision["answer"] or decision["refusal_reason"]
            evaluation = ({"status": "skipped", "reason": "generation_unavailable", "evaluation_version": EVALUATION_VERSION, "abstention_recommended": True, "evaluation_latency_ms": 0.0}
                          if decision.get("_legacy_status") == "generation_unavailable" else await asyncio.to_thread(_evaluate, question, answer, contexts, decision["decision"] == "ANSWER"))
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
