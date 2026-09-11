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
from .pipeline import LLMUnavailableError, NewsPipeline, PIPELINE_VERSION
from .evaluation import EVALUATION_VERSION, evaluate_response
from .evidence_router import ANSWER, INSUFFICIENT, REFUSE, SINGLE_DOC
from .request_trace import finish_trace, start_trace, trace_phase

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
        "config | embedding_model=%s | embedding_device=%s",
        config.EMBEDDING_MODEL, config.EMBEDDING_DEVICE
    )
    LOGGER.info(
        "config | reranker_model=%s | reranker_batch=%d | rearank_max_length=%d | rerank_min_score=%.2f | rerank_min_margin=%.2f | rerank_partial_min_score=%.2f | rerank_evidence_chunk_delta=%.2f",
        config.RERANKER_MODEL, config.RERANK_BATCH_SIZE, config.RERANK_MAX_LENGTH, config.RERANK_MIN_SCORE, config.RERANK_MIN_MARGIN, config.RERANK_PARTIAL_MIN_SCORE, config.RERANK_EVIDENCE_CHUNK_DELTA,
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


def _fallback_generation_plan(contexts: list[dict], sufficient: bool) -> dict:
    """Keep old pipeline doubles usable while exposing additive decision fields."""
    route = SINGLE_DOC if sufficient else INSUFFICIENT
    return {
        "evidence_plan": {},
        "coverage_matrix": [],
        "route_decision": {
            "route": route,
            "reason": "legacy_evidence_gate",
            "covered_sub_questions": [],
            "missing_sub_questions": [] if sufficient else ["question evidence"],
            "selected_article_ids": [str(item.get("article_id")) for item in contexts if item.get("article_id")] if sufficient else [],
        },
    }


def _route_decision(plan: dict) -> dict:
    return plan.get("route_decision") if isinstance(plan.get("route_decision"), dict) else {}


def _should_answer(plan: dict) -> bool:
    return _route_decision(plan).get("route") in {SINGLE_DOC, "REQUIRES_MULTI_DOC"}


def _generation_plan(question: str, contexts: list[dict], sufficient: bool) -> dict:
    planner = getattr(pipeline, "plan_evidence", None)
    if not callable(planner):
        return _fallback_generation_plan(contexts, sufficient)
    try:
        result = planner(question, contexts)
        if not isinstance(result, dict) or not isinstance(result.get("route_decision"), dict):
            return _fallback_generation_plan(contexts, False)
        return result
    except Exception:
        LOGGER.exception("evidence planning failed; refusing")
        return {
            "evidence_plan": {},
            "coverage_matrix": [],
            "route_decision": {"route": INSUFFICIENT, "reason": "evidence_plan_unavailable", "covered_sub_questions": [], "missing_sub_questions": [], "selected_article_ids": []},
        }


def _generation_contexts(contexts: list[dict], plan: dict) -> list[dict]:
    """Keep complete evidence chunks from selected articles for final generation."""
    selected = getattr(pipeline, "last_planned_contexts", None)
    if isinstance(selected, list) and selected and _should_answer(plan):
        selected_sources = {
            source for item in selected
            if (source := str(item.get("article_id") or item.get("url") or ""))
        }
        complete_contexts = [
            item for item in contexts
            if str(item.get("article_id") or item.get("url") or "") in selected_sources
        ]
        if complete_contexts:
            return complete_contexts
        return selected
    return contexts


def _decision_payload(plan: dict, answer: str, contexts: list[dict]) -> dict:
    """Normalize additive generation decision schema."""
    citations = [
        {"article_id": item.get("article_id"), "title": item.get("title"),
         "url": item.get("url"), "score": item.get("rerank_score")}
        for item in contexts
    ]
    return {
        "decision": ANSWER if _should_answer(plan) else REFUSE,
        "answer": answer,
        "citations": citations,
    }


def _answer_mode_for_trace(question: str) -> str:
    chooser = getattr(pipeline, "_answer_mode", None)
    try:
        mode = chooser(question) if callable(chooser) else None
    except Exception:
        mode = None
    return str(mode or "DIRECT")

@app.post("/api/qa/ask")
@app.post("/ask")
def ask(request: AskRequest):
    started = time.perf_counter()
    trace, trace_token = start_trace(
        endpoint="POST /api/qa/ask", question=request.question, top_k=request.top_k,
    )
    try:
        LOGGER.info("request start | question_chars=%d | top_k=%d", len(request.question), request.top_k)
        contexts, sufficient, top_score, margin = pipeline.search_with_evidence(request.question, request.top_k)
        plan = _generation_plan(request.question, contexts, sufficient)
        generation_contexts = _generation_contexts(contexts, plan)
        generation_error_class = None
        answer_mode = _answer_mode_for_trace(request.question)
        if _should_answer(plan):
            try:
                answer = pipeline.generate(request.question, generation_contexts)
                answer_status = "generated"
            except LLMUnavailableError as exc:
                generation_error_class = type(exc).__name__
                fallback = getattr(pipeline, "extractive_fallback", None)
                answer = (fallback(request.question, generation_contexts)
                          if callable(fallback) else "Không thể tạo câu trả lời từ mô hình lúc này; dữ liệu vẫn đủ bằng chứng nhưng hệ thống không nhận được đầu ra hợp lệ.")
                answer_status = "extractive_fallback" if callable(fallback) else "generation_unavailable"
                LOGGER.error("generation unavailable; returning extractive fallback=%s", callable(fallback))
        else:
            answer = "Không đủ thông tin trong dữ liệu được cung cấp để trả lời câu hỏi này một cách đáng tin cậy."
            answer_status = "abstained"
            LOGGER.warning(
                "abstention | route=%s | reason=%s | missing=%s | top_score=%.4f | margin=%.4f | min_score=%.2f | min_margin=%.2f",
                _route_decision(plan).get("route"), _route_decision(plan).get("reason"),
                _route_decision(plan).get("missing_sub_questions", []),
                top_score, margin, config.RERANK_MIN_SCORE, config.RERANK_MIN_MARGIN,
            )
        evaluation = ({"status": "skipped", "reason": "generation_unavailable", "evaluation_version": EVALUATION_VERSION, "abstention_recommended": True, "evaluation_latency_ms": 0.0}
                      if answer_status == "generation_unavailable" else _evaluate(request.question, answer, generation_contexts, _should_answer(plan)))
        trace_phase("generation_decision", {
            "answer_status": answer_status,
            "answer_mode": answer_mode,
            "pipeline_version": PIPELINE_VERSION,
            "generation_error_class": generation_error_class,
            "route_decision": _route_decision(plan),
            "generation_contexts": generation_contexts,
            "answer": answer,
        })
        trace_phase("evaluation", evaluation)
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
            "answer_mode": answer_mode,
            "pipeline_version": PIPELINE_VERSION,
            "generation_error_class": generation_error_class,
            "generation_decision": _decision_payload(plan, answer, generation_contexts),
            "evidence_plan": plan.get("evidence_plan", {}),
            "coverage_matrix": plan.get("coverage_matrix", []),
            "route_decision": _route_decision(plan),
            "response_time_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        LOGGER.info("request done | contexts=%d | response_ms=%.1f", len(contexts), payload["response_time_ms"])
        finish_trace(trace, trace_token, status=answer_status, payload=payload)
        return payload
    except Exception as exc:
        trace_phase("error", {"error_type": type(exc).__name__, "message": str(exc)})
        finish_trace(trace, trace_token, status="error", payload={"error_type": type(exc).__name__})
        raise


async def stream_answer(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            body = json.loads(await websocket.receive_text())
            question = str(body.get("text") or body.get("question") or "").strip()
            if not question:
                continue
            contexts, sufficient, top_score, margin = await asyncio.to_thread(pipeline.search_with_evidence, question, 5)
            plan = _generation_plan(question, contexts, sufficient)
            generation_contexts = _generation_contexts(contexts, plan)
            if _should_answer(plan):
                try:
                    answer = await asyncio.to_thread(pipeline.generate, question, generation_contexts)
                    answer_status = "generated"
                except LLMUnavailableError:
                    fallback = getattr(pipeline, "extractive_fallback", None)
                    answer = (fallback(question, generation_contexts)
                              if callable(fallback) else "Không thể tạo câu trả lời từ mô hình lúc này; hệ thống không nhận được đầu ra hợp lệ.")
                    answer_status = "extractive_fallback" if callable(fallback) else "generation_unavailable"
                    LOGGER.error("stream generation unavailable; returning extractive fallback=%s", callable(fallback))
            else:
                answer = "Không đủ thông tin trong dữ liệu được cung cấp để trả lời câu hỏi này một cách đáng tin cậy."
                answer_status = "abstained"
                LOGGER.warning("stream abstention | top_score=%.4f | margin=%.4f", top_score, margin)
            evaluation = ({"status": "skipped", "reason": "generation_unavailable", "evaluation_version": EVALUATION_VERSION, "abstention_recommended": True, "evaluation_latency_ms": 0.0}
                          if answer_status == "generation_unavailable" else await asyncio.to_thread(_evaluate, question, answer, generation_contexts, _should_answer(plan)))
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


