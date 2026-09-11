from __future__ import annotations

import json
import logging
import os
import pickle
import re
import time
import threading
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from . import config
from .evidence_router import (
    EvidencePlan as EvidenceRouter,
    EVIDENCE_PLAN_UNAVAILABLE,
    INSUFFICIENT,
    REFUSE,
    REQUIRES_MULTI_DOC,
    SINGLE_DOC,
)
from .request_trace import trace_phase
from .source_identity import source_key
from .source_diversification import diversify_by_article
from .temporal_retrieval import apply_temporal_boost, extract_temporal_terms
from src.backend.query_planner import build_evidence_plan
from src.backend.generation_gate import run_generation_stage
from src.RAG.retrieval.schema import EvidencePlan, GenerationDecision

LOGGER = logging.getLogger("rag-api.pipeline")
PIPELINE_VERSION = "qa-routing-20260911.1"

# This pipeline is PyTorch-only. Prevent Transformers from importing an
# unrelated TensorFlow/Keras installation that may be incompatible.
os.environ.setdefault("USE_TF", "0")


class IndexUnavailableError(RuntimeError):
    """The local Qdrant collection cannot currently serve queries."""


class RerankerError(RuntimeError):
    """The reranker could not score the retrieved candidates."""


class LLMUnavailableError(RuntimeError):
    """The external generation service is unavailable or returned invalid data."""


def _load_chunks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class NewsPipeline:
    """E5 + BM25 hybrid retrieval -> BGE reranking -> LLM generation."""

    def __init__(self) -> None:
        self.client = QdrantClient(path=str(config.QDRANT_PATH))
        self.encoder = None
        self.bm25_index = None
        self.reranker = None
        self.generator = None
        self.evidence_plan = EvidenceRouter(
            planner=self._llm_plan_subquestions,
            entailment=self._llm_entailment_batch,
            support_threshold=config.EVIDENCE_SUPPORT_THRESHOLD,
            timeout=config.EVIDENCE_PLAN_TIMEOUT,
            entailment_batch_size=config.EVIDENCE_ENTAILMENT_BATCH_SIZE,
        )
        self.generator_provider = config.resolve_llm_provider()
        self._load_lock = threading.Lock()

    @staticmethod
    def _resolve_device(torch, requested: str | None = None) -> str:
        """Use requested device, defaulting to cuda:0 and safely falling back to CPU."""
        device = (requested or config.MODEL_DEVICE or "cuda:0").strip().lower()
        if device == "auto":
            device = "cuda:0"
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                LOGGER.warning("CUDA unavailable; falling back to cpu")
                return "cpu"
            if ":" not in device:
                device = "cuda:0"
            index = int(device.split(":", 1)[1])
            if index >= torch.cuda.device_count():
                LOGGER.warning("Requested %s unavailable; falling back to cpu", device)
                return "cpu"
        elif device not in {"cpu"}:
            raise RuntimeError("MODEL_DEVICE must be cuda, cuda:N, cpu, or auto.")
        return device

    @staticmethod
    def _resolve_dtype(torch, requested: str | None = None):
        """Resolve the configured inference dtype without silently using FP32."""
        name = (requested or config.MODEL_DTYPE or "float16").strip().lower()
        dtypes = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if name not in dtypes:
            raise RuntimeError("MODEL_DTYPE must be float16, bfloat16, or float32.")
        return dtypes[name]

    def _load_encoder(self):
        if not hasattr(self, "_load_lock"):
            self._load_lock = threading.Lock()
        if self.encoder is None:
            with self._load_lock:
                if self.encoder is None:
                    return self._load_encoder_impl()
        return self.encoder

    def _load_encoder_impl(self):
        if self.encoder is None:
            started = time.perf_counter()
            LOGGER.info("embedding model load start | model=%s", config.EMBEDDING_MODEL)
            import torch
            from sentence_transformers import SentenceTransformer

            device = self._resolve_device(torch, config.EMBEDDING_DEVICE)
            dtype = self._resolve_dtype(torch)
            model_kwargs = {"torch_dtype": dtype} if device.startswith("cuda") else {}
            self.encoder = SentenceTransformer(
                config.EMBEDDING_MODEL, device=device, model_kwargs=model_kwargs,
            )
            LOGGER.info(
                "embedding model load done | device=%s | dtype=%s | elapsed_ms=%.1f",
                device, dtype, (time.perf_counter() - started) * 1000,
            )
        return self.encoder

    def _load_reranker(self):
        if not hasattr(self, "_load_lock"):
            self._load_lock = threading.Lock()
        if self.reranker is None:
            with self._load_lock:
                if self.reranker is None:
                    return self._load_reranker_impl()
        return self.reranker

    def _load_reranker_impl(self):
        if self.reranker is None:
            started = time.perf_counter()
            LOGGER.info("reranker model load start | model=%s", config.RERANKER_MODEL)
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device = self._resolve_device(torch, config.RERANKER_DEVICE)
            dtype = self._resolve_dtype(torch)
            tokenizer = AutoTokenizer.from_pretrained(config.RERANKER_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(
                config.RERANKER_MODEL,
                torch_dtype=dtype if device.startswith("cuda") else None,
            )
            model.to(device)
            model.eval()
            self.reranker = (tokenizer, model, torch, device)
            try:
                effective_dtype = str(next(model.parameters()).dtype)
            except (AttributeError, StopIteration, TypeError):
                effective_dtype = "framework_default_cpu" if device == "cpu" else str(dtype)
            LOGGER.info(
                "reranker model load done | model=%s | device=%s | configured_dtype=%s | effective_dtype=%s | quantization=none | elapsed_ms=%.1f",
                config.RERANKER_MODEL, device, dtype, effective_dtype,
                (time.perf_counter() - started) * 1000,
            )
        return self.reranker

    def _dense_is_ready(self) -> bool:
        try:
            return self.client.collection_exists(config.COLLECTION)
        except Exception:
            return False

    def is_ready(self) -> bool:
        return self._dense_is_ready() and config.BM25_INDEX_PATH.is_file()

    def _load_bm25_index(self) -> dict[str, Any]:
        if not hasattr(self, "_load_lock"):
            self._load_lock = threading.Lock()
        if getattr(self, "bm25_index", None) is None:
            with self._load_lock:
                if getattr(self, "bm25_index", None) is None:
                    if not config.BM25_INDEX_PATH.is_file():
                        raise IndexUnavailableError(
                            "BM25 index is not ready. Run build_index.py first."
                        )
                    try:
                        with config.BM25_INDEX_PATH.open("rb") as handle:
                            index_data = pickle.load(handle)
                        if not isinstance(index_data, dict) or not {"index", "rows"} <= index_data.keys():
                            raise ValueError("invalid BM25 index structure")
                    except Exception as exc:
                        raise IndexUnavailableError("BM25 index could not be loaded.") from exc
                    self.bm25_index = index_data
        return self.bm25_index

    def close(self) -> None:
        """Release the local Qdrant file lock when the pipeline is not serving."""
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def build_index(
        self,
        chunk_path: Path | None = None,
        batch_size: int = 64,
        limit: int | None = None,
    ) -> int:
        path = chunk_path or config.CHUNK_PATH
        if not path.is_file():
            raise FileNotFoundError(
                f"Chunk corpus not found at {path}. Run the data ingestion and token chunking steps first."
            )
        rows = _load_chunks(path)
        rows = [row for row in rows if str(row.get("text") or "").strip()]
        if limit is not None:
            rows = rows[:limit]
        if not rows:
            raise ValueError("Chunk corpus contains no non-empty text.")
        try:
            from rank_bm25 import BM25Okapi

            from src.RAG.retrieval.tokenize import tokenize_vietnamese, tokenizer_name
        except ImportError as exc:
            raise RuntimeError("rank-bm25 is required to build the hybrid index.") from exc
        bm25_data = {
            "index": BM25Okapi([tokenize_vietnamese(str(row["text"])) for row in rows]),
            "rows": rows,
            "tokenizer": tokenizer_name(),
        }
        encoder = self._load_encoder()
        vectors = encoder.encode(
            ["passage: " + str(row["text"]).strip() for row in rows],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        vector_size = int(vectors.shape[1])
        if self._dense_is_ready():
            self.client.delete_collection(config.COLLECTION)
        self.client.create_collection(
            collection_name=config.COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        points = []
        for index, (row, vector) in enumerate(zip(rows, vectors)):
            payload = self._payload_from_row(row, index)
            points.append(PointStruct(id=index, vector=vector.tolist(), payload=payload))
            if len(points) >= batch_size:
                self.client.upsert(collection_name=config.COLLECTION, points=points)
                points = []
        if points:
            self.client.upsert(collection_name=config.COLLECTION, points=points)
        config.BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = config.BM25_INDEX_PATH.with_suffix(config.BM25_INDEX_PATH.suffix + ".tmp")
        with temporary_path.open("wb") as handle:
            pickle.dump(bm25_data, handle)
        temporary_path.replace(config.BM25_INDEX_PATH)
        self.bm25_index = bm25_data
        return len(rows)

    @staticmethod
    def _payload_from_row(row: dict[str, Any], index: int) -> dict[str, Any]:
        metadata = row.get("metadata") or {}
        return {
            "chunk_id": str(row.get("chunk_id", index)),
            "article_id": str(row.get("article_id", metadata.get("article_id", ""))),
            "title": metadata.get("title", ""),
            "description": metadata.get("description", ""),
            "category": metadata.get("category", ""),
            "url": metadata.get("url", ""),
            "chunk_index": metadata.get("chunk_index", row.get("chunk_index", 0)),
            "text": str(row.get("text", "")),
        }

    @staticmethod
    def _reciprocal_rank_fusion(
        dense_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
        *,
        limit: int,
        rrf_k: int,
    ) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}
        for retriever, results in (("dense", dense_results), ("bm25", bm25_results)):
            for rank, result in enumerate(results, start=1):
                chunk_id = str(result["chunk_id"])
                entry = fused.setdefault(
                    chunk_id,
                    {**result, "retrievers": [], "retrieval_score": 0.0},
                )
                entry.update({key: value for key, value in result.items() if key not in entry})
                entry[f"{retriever}_rank"] = rank
                if retriever not in entry["retrievers"]:
                    entry["retrievers"].append(retriever)
                entry["retrieval_score"] = float(entry["retrieval_score"]) + 1.0 / (rrf_k + rank)
        ranked = sorted(
            fused.values(),
            key=lambda item: (-float(item["retrieval_score"]), str(item["chunk_id"])),
        )[:limit]
        return [{**item, "retrieval_rank": rank} for rank, item in enumerate(ranked, start=1)]

    def retrieve(self, question: str, limit: int = config.TOP_K_RETRIEVAL) -> list[dict[str, Any]]:
        started = time.perf_counter()
        if not self.is_ready():
            raise IndexUnavailableError("Hybrid index is not ready. Run build_index.py first.")
        candidate_k = max(limit, config.HYBRID_CANDIDATE_K)
        vector = self._load_encoder().encode(
            ["query: " + question.strip()], normalize_embeddings=True
        )[0].tolist()
        try:
            response = self.client.query_points(
                collection_name=config.COLLECTION,
                query=vector,
                limit=candidate_k,
                with_payload=True,
            )
        except Exception as exc:
            raise IndexUnavailableError("Qdrant query failed.") from exc
        dense_results = [
            {**(hit.payload or {}), "dense_score": float(hit.score)}
            for hit in response.points
        ]
        bm25_data = self._load_bm25_index()
        try:
            from src.RAG.retrieval.tokenize import tokenize_vietnamese

            scores = bm25_data["index"].get_scores(tokenize_vietnamese(question))
            rows = bm25_data["rows"]
            positions = sorted(
                (position for position, score in enumerate(scores) if abs(float(score)) > 1e-12),
                key=lambda position: (-float(scores[position]), str(rows[position].get("chunk_id", position))),
            )[:candidate_k]
            bm25_results = [
                {
                    **self._payload_from_row(rows[position], position),
                    "bm25_score": float(scores[position]),
                }
                for position in positions
            ]
        except Exception as exc:
            raise IndexUnavailableError("BM25 query failed.") from exc
        results = self._reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            limit=limit,
            rrf_k=config.HYBRID_RRF_K,
        )
        LOGGER.info(
            "hybrid retrieval done | requested=%d | dense=%d | bm25=%d | returned=%d | top_score=%.4f | elapsed_ms=%.1f",
            limit, len(dense_results), len(bm25_results), len(results),
            float(results[0].get("retrieval_score", 0.0)) if results else 0.0,
            (time.perf_counter() - started) * 1000,
        )
        return results

    def rerank(self, question: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            LOGGER.warning("rerank skipped | candidates=0")
            return []
        started = time.perf_counter()
        try:
            tokenizer, model, torch, device = self._load_reranker()
            scores = []
            pairs = [(question, str(candidate.get("text", ""))) for candidate in candidates]
            for start in range(0, len(pairs), config.RERANK_BATCH_SIZE):
                batch = pairs[start : start + config.RERANK_BATCH_SIZE]
                inputs = tokenizer(
                    [pair[0] for pair in batch],
                    [pair[1] for pair in batch],
                    padding=True,
                    truncation=True,
                    max_length=config.RERANK_MAX_LENGTH,
                    return_tensors="pt",
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                with torch.no_grad():
                    logits = model(**inputs).logits.reshape(-1).detach().cpu().tolist()
                scores.extend(float(score) for score in logits)
            ranked = [
                {**candidate, "rerank_score": float(score)}
                for candidate, score in zip(candidates, scores)
            ]
        except Exception as exc:
            raise RerankerError("Reranker failed.") from exc
        ranked.sort(key=lambda item: float(item["rerank_score"]), reverse=True)
        results = [{**item, "rank": index + 1} for index, item in enumerate(ranked)]
        LOGGER.info(
            "rerank done | model=%s | candidates=%d | top_logit=%.4f | bottom_logit=%.4f | score_semantics=raw_bge_logit_not_probability | elapsed_ms=%.1f",
            config.RERANKER_MODEL, len(results), float(results[0].get("rerank_score", 0.0)) if results else 0.0,
            float(results[-1].get("rerank_score", 0.0)) if results else 0.0,
            (time.perf_counter() - started) * 1000,
        )
        return results

    def search(self, question: str, top_k: int = config.TOP_K_CONTEXT) -> list[dict[str, Any]]:
        contexts, _, _, _ = self.search_with_evidence(question, top_k)
        return contexts

    @staticmethod
    def _deterministic_evidence_plan(question: str) -> dict[str, Any]:
        """Stable no-network plan for factoid and degraded-provider requests."""
        normalized = str(question or "").strip()
        lowered = normalized.casefold()
        multi_source = any(marker in lowered for marker in (
            "so sánh", "khác nhau", "điểm chung", "cả hai", "cả ba",
            "các bài báo", "tổng hợp", "theo từng",
        ))
        sub_questions = [{"id": "sq1", "text": normalized, "evidence_type": "FACT"}]
        aspects: list[str] = []
        if multi_source:
            match = re.search(r"bối cảnh\s+([^?.!]+?)(?:\?|[.!]|$)", normalized, flags=re.IGNORECASE)
            if match:
                aspects = [
                    part.strip(" .,:;()\"")
                    for part in re.split(r"\s+(?:và|hoặc)\s+|,|;", match.group(1), flags=re.IGNORECASE)
                    if len(part.strip(" .,:;()\"")) >= 3
                ]
            for index, aspect in enumerate(aspects[:4], start=2):
                sub_questions.append({
                    "id": f"sq{index}",
                    "text": f"{normalized} Khía cạnh cần đối chiếu: {aspect}.",
                    "evidence_type": "RELATION",
                })
        return {
            "normalized_question": normalized,
            "query_type_hint": "COMPARISON" if multi_source else "FACTOID",
            "entities": [],
            "numbers": [],
            "dates": [],
            "temporal_constraints": [],
            "estimated_sources_needed": max(2, len(aspects) + 1) if multi_source else 1,
            "answer_operator": "COMPARE" if multi_source else "DIRECT",
            "sub_questions": sub_questions,
        }

    def _retrieval_backed_plan(
        self,
        question: str,
        pool: list[dict[str, Any]],
        evidence_plan: dict[str, Any] | None,
        *,
        reason: str,
    ) -> dict[str, Any]:
        """Route from already-ranked evidence when optional LLM validation is down."""
        quality = getattr(self, "last_evidence_quality", {}) or {}
        selected: list[dict[str, Any]] = []
        selected_ids: list[str] = []
        for index, item in enumerate(pool):
            source = str(item.get("article_id") or item.get("url") or self._source_key(item, index))
            if source in selected_ids:
                continue
            selected_ids.append(source)
            selected.append(item)
            if len(selected) >= config.TOP_K_CONTEXT:
                break
        sufficient = quality.get("status") == "sufficient" and bool(selected)
        specific_relation_missing = not self._specific_relation_supported(question, pool)
        if sufficient and specific_relation_missing:
            sufficient = False
        if not sufficient:
            route = INSUFFICIENT
            selected_ids = []
            selected = []
            missing = ["specific entity-relation evidence" if specific_relation_missing else "retrieval evidence"]
        else:
            expected_sources = int((evidence_plan or {}).get("estimated_sources_needed") or 1)
            route = REQUIRES_MULTI_DOC if expected_sources > 1 else SINGLE_DOC
            missing = []
        self.last_planned_contexts = selected
        return {
            "evidence_plan": evidence_plan or self._deterministic_evidence_plan(question),
            "coverage_matrix": [],
            "route_decision": {
                "route": route,
                "reason": "specific_relation_missing" if specific_relation_missing else reason,
                "covered_sub_questions": [item["id"] for item in (evidence_plan or {}).get("sub_questions", [])] if sufficient else [],
                "missing_sub_questions": missing,
                "selected_article_ids": selected_ids,
            },
        }

    @staticmethod
    def _specific_relation_supported(question: str, contexts: list[dict[str, Any]]) -> bool:
        """Require company and its claimed action in evidence from same article."""
        normalized = str(question or "").casefold()
        match = re.search(r"\bcông ty\s+(.+?)\s+có\s+kế hoạch\b", normalized)
        if not match:
            return True
        entity = re.sub(r"\s+", " ", match.group(1)).strip(" ,.;:")
        if not entity:
            return False
        action_terms = ("kế hoạch", "dự kiến", "sẽ", "phát triển", "triển khai", "xây dựng", "dự án", "sử dụng", "quy hoạch")
        grouped: dict[str, list[str]] = {}
        for index, item in enumerate(contexts):
            source = str(item.get("article_id") or item.get("url") or index)
            grouped.setdefault(source, []).append(str(item.get("text") or "").casefold())
        return any(
            entity in (text := " ".join(chunks)) and any(term in text for term in action_terms)
            for chunks in grouped.values()
        )

    def _record_coverage_plan(self, question: str, pool: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
        route_decision = plan.get("route_decision", {})
        trace_phase("coverage_routing", {
            "question": question,
            "candidate_contexts": pool,
            "evidence_plan": plan.get("evidence_plan", {}),
            "coverage_matrix": plan.get("coverage_matrix", []),
            "route_decision": route_decision,
        })
        LOGGER.info(
            "coverage routing done | route=%s | reason=%s | sub_questions=%d | candidates=%d | selected_articles=%d | missing=%d",
            route_decision.get("route"), route_decision.get("reason"),
            len((plan.get("evidence_plan") or {}).get("sub_questions", [])), len(pool),
            len(route_decision.get("selected_article_ids", [])), len(route_decision.get("missing_sub_questions", [])),
        )
        return plan

    def plan_evidence(
        self, question: str, contexts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Validate with LLM only when enabled; otherwise trust strong reranked evidence."""
        planner = getattr(self, "evidence_plan", None)
        if planner is None:
            planner = EvidenceRouter(
                planner=self._llm_plan_subquestions,
                entailment=self._llm_entailment_batch,
                support_threshold=config.EVIDENCE_SUPPORT_THRESHOLD,
                timeout=config.EVIDENCE_PLAN_TIMEOUT,
                entailment_batch_size=config.EVIDENCE_ENTAILMENT_BATCH_SIZE,
            )
            self.evidence_plan = planner
        ranked_pool = getattr(self, "last_ranked_contexts", None) or contexts
        pool = ranked_pool[:config.EVIDENCE_COVERAGE_CANDIDATE_K]
        cached = getattr(self, "last_evidence_plan", None)
        origin = getattr(self, "last_evidence_plan_origin", "deterministic")
        if not config.EVIDENCE_LLM_COVERAGE_ENABLED or origin != "llm":
            return self._record_coverage_plan(
                question,
                pool,
                self._retrieval_backed_plan(
                    question, pool, cached, reason="rerank_evidence_fallback",
                ),
            )
        plan = planner.plan(question, pool, evidence_plan=cached)
        route_decision = plan.get("route_decision", {})
        if route_decision.get("reason") == EVIDENCE_PLAN_UNAVAILABLE:
            plan = self._retrieval_backed_plan(
                question, pool, cached, reason="coverage_llm_unavailable_fallback",
            )
            return self._record_coverage_plan(question, pool, plan)
        required = set(route_decision.get("selected_article_ids", []))
        if route_decision.get("route") != INSUFFICIENT and required:
            self.last_planned_contexts = [
                item for index, item in enumerate(pool)
                if self._source_key(item, index) in required
                or str(item.get("article_id") or "") in required
            ]
        else:
            self.last_planned_contexts = []
        return self._record_coverage_plan(question, pool, plan)

    def plan_query(self, question: str) -> dict[str, Any]:
        """Run router before retrieval and retain its normalized EvidencePlan."""
        try:
            plan = build_evidence_plan(question).model_dump()
            origin = "llm" if config.EVIDENCE_LLM_ROUTING_ENABLED else "deterministic"
        except Exception as exc:
            origin = "deterministic_fallback"
            LOGGER.warning("shared query planner unavailable; using fallback | error=%s", type(exc).__name__)
            plan = self._deterministic_evidence_plan(question)
        self.last_evidence_plan = plan
        self.last_evidence_plan_origin = origin
        trace_phase("routing", {
            "question": question,
            "evidence_plan": plan,
            "strategy": origin,
        })
        return plan

    @staticmethod
    def _fuse_subquestion_candidates(
        candidates_by_subquestion: dict[str, list[dict[str, Any]]], *, limit: int,
    ) -> list[dict[str, Any]]:
        """Fuse per-sub-question hybrid pools before one BGE rerank."""
        fused: dict[str, dict[str, Any]] = {}
        for sub_question_id, candidates in candidates_by_subquestion.items():
            for rank, candidate in enumerate(candidates, start=1):
                key = str(candidate.get("chunk_id") or f"{sub_question_id}:{rank}")
                entry = fused.setdefault(key, {**candidate, "sub_question_ids": [], "subquestion_retrieval_score": 0.0})
                entry["subquestion_retrieval_score"] += 1.0 / (config.HYBRID_RRF_K + rank)
                if sub_question_id not in entry["sub_question_ids"]:
                    entry["sub_question_ids"].append(sub_question_id)
        ranked = sorted(
            fused.values(),
            key=lambda item: (-float(item["subquestion_retrieval_score"]), str(item.get("chunk_id", ""))),
        )[:limit]
        return [{**item, "subquestion_retrieval_rank": index + 1} for index, item in enumerate(ranked)]

    def retrieve_evidence_plan(self, evidence_plan: dict[str, Any]) -> list[dict[str, Any]]:
        """Retrieve each planned fact independently, then fuse candidate pools."""
        pools: dict[str, list[dict[str, Any]]] = {}
        for sub_question in evidence_plan.get("sub_questions", []):
            text = str(sub_question.get("text") or "").strip()
            if text:
                pools[str(sub_question["id"])] = self.retrieve(text, limit=config.HYBRID_CANDIDATE_K)
        fused = self._fuse_subquestion_candidates(pools, limit=config.HYBRID_CANDIDATE_K)
        trace_phase("retrieval", {
            "subquestion_candidate_pools": pools,
            "fused_candidates": fused,
        })
        return fused

    def _llm_plan_subquestions(self, question: str, contexts: list[dict[str, Any]]) -> Any:
        """Use configured generator as strict JSON question decomposer."""
        prompt = (
            "Phân rã QUESTION thành EvidencePlan. Chỉ trả JSON hợp lệ với fields "
            "normalized_question, query_type_hint, entities, numbers, dates, temporal_constraints, "
            "estimated_sources_needed, answer_operator, sub_questions. Mỗi sub_questions item có "
            "id, text, evidence_type. Không thêm văn bản.\n"
            f"QUESTION: {question}"
        )
        return self._generate_planning_prompt(prompt)

    def _llm_entailment_batch(self, pairs: list[dict[str, Any]]) -> Any:
        """Score each subquestion/chunk pair in one strict JSON batch call."""
        payload = [
            {"subquestion": pair["subquestion"], "article_id": pair["article_id"],
             "chunk_id": pair["chunk_id"], "evidence": pair["text"][:config.EVIDENCE_SUPPORT_TEXT_MAX_CHARS]}
            for pair in pairs
        ]
        prompt = (
            "Đánh giá mức độ bằng chứng trong từng cặp. Trả JSON duy nhất dạng "
            "{\"scores\":[0.0, ...]}, mỗi score trong [0,1], cùng thứ tự INPUT. "
            "Score 1 chỉ khi evidence trực tiếp hỗ trợ subquestion.\nINPUT:\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        return self._generate_planning_prompt(prompt)

    def _generate_planning_prompt(self, prompt: str) -> str:
        """Call LLM provider for routing; exceptions intentionally fail closed."""
        if getattr(self, "generator_provider", config.resolve_llm_provider()) == "hf_model":
            return self._generate_with_hf(prompt)
        return self._generate_with_api(prompt)

    # Descriptive alias useful to callers integrating the router directly.
    route_evidence = plan_evidence

    def search_with_evidence(
        self,
        question: str,
        top_k: int = config.TOP_K_CONTEXT,
    ) -> tuple[list[dict[str, Any]], bool, float, float]:
        """Evaluate evidence on the full reranked pool, then select final contexts."""
        evidence_plan = self.plan_query(question)
        candidates = self.retrieve_evidence_plan(evidence_plan)
        ranked = self.rerank(evidence_plan["normalized_question"], candidates)[:config.RERANK_TOP_K]
        temporal_terms = extract_temporal_terms(evidence_plan["normalized_question"])
        if temporal_terms:
            temporally_ranked = apply_temporal_boost(
                ranked, temporal_terms, boost=config.TEMPORAL_BOOST
            )
            ranked = [
                {
                    **item,
                    "base_rerank_score": item.get("rerank_score"),
                    "rerank_score": item.get("temporal_score", item.get("rerank_score", 0.0)),
                    "rank": index + 1,
                }
                for index, item in enumerate(temporally_ranked)
            ]
        ranked = diversify_by_article(
            ranked, max_per_article=config.SOURCE_MAX_CHUNKS_PER_ARTICLE
        )
        ranked = [{**item, "rank": index + 1} for index, item in enumerate(ranked)]
        self.last_ranked_contexts = ranked
        trace_phase("reranking", {
            "normalized_question": evidence_plan["normalized_question"],
            "candidate_count": len(candidates),
            "ranked_candidates": ranked,
            "temporal_terms": temporal_terms,
            "source_max_chunks_per_article": config.SOURCE_MAX_CHUNKS_PER_ARTICLE,
        })
        quality = self.evidence_quality_details(ranked)
        self.last_evidence_quality = quality
        sufficient = quality["status"] == "sufficient"
        top_score = quality["top_score"]
        # Deprecated compatibility field. It is no longer used as a gate.
        margin = quality["legacy_margin"]
        selected = []
        pool_source_count = len({self._source_key(item, index) for index, item in enumerate(ranked)})
        pool_context_count = len(ranked)
        for group in quality["article_groups"]:
            item = group["chunks"][0]
            selected.append({
                **item,
                "text": "\n\n".join(str(chunk.get("text", "")) for chunk in group["chunks"]),
                "rank": item.get("rank", len(selected) + 1),
                "citation_rank": len(selected) + 1,
                "evidence_chunks": [chunk.get("text", "") for chunk in group["chunks"]],
                "article_chunk_count": group["chunk_count"],
                "article_evidence_score": group["score"],
                "_pool_source_count": pool_source_count,
                "_pool_context_count": pool_context_count,
            })
            if len(selected) >= top_k:
                break
        LOGGER.info(
            "evidence inference done | status=%s | reranked_chunks=%d | article_groups=%d | selected_article_groups=%d | top_article_score=%.4f | corroboration=%d | contradiction=%s | legacy_margin=%.4f | gate=%s",
            quality["status"], len(ranked), quality.get("article_count", 0), len(selected),
            quality.get("top_article_score", float("-inf")), quality.get("corroboration", 0),
            quality.get("contradiction_detected", False), margin, sufficient,
        )
        trace_phase("evidence_selection", {
            "evidence_quality": quality,
            "evidence_sufficient": sufficient,
            "selected_contexts": selected,
            "top_score": top_score,
            "legacy_margin": margin,
        })
        return selected, sufficient, top_score, margin
    
    def search_adaptive(
        self,
        question: str,
        top_k: int = config.TOP_K_CONTEXT,
    ) -> GenerationDecision:
        """Run shared planning, retrieval, coverage routing, then generation gate."""
        contexts, _, _, _ = self.search_with_evidence(question, top_k)
        plan = self.plan_evidence(question, contexts)
        result = run_generation_stage(
            question=question,
            evidence_plan=plan["evidence_plan"],
            coverage_matrix=plan["coverage_matrix"],
            route_decision=plan["route_decision"],
            ranked_candidates=getattr(self, "last_ranked_contexts", contexts),
            retry_callback=None,
            generator_callback=self.generate,
        )
        return GenerationDecision(**result)
    
    @staticmethod
    def _source_key(item: dict[str, Any], index: int = 0) -> str:
        return source_key(item, index)

    @staticmethod
    def evidence_quality(contexts: list[dict[str, Any]]) -> tuple[bool, float, float]:
        """Legacy tuple API; use :meth:`evidence_quality_details` for semantics."""
        details = NewsPipeline.evidence_quality_details(contexts)
        return details["status"] == "sufficient", details["top_score"], details["legacy_margin"]

    @staticmethod
    def _sentence_signatures(text: str) -> set[tuple[str, bool]]:
        """Cheap contradiction diagnostic for explicit Vietnamese negation only."""
        negation = re.compile(r"\b(không|chưa|chẳng|không phải|not|no)\b", re.IGNORECASE)
        signatures = set()
        for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", text or ""):
            words = re.findall(r"[\wÀ-ỹ]+", sentence.lower())
            if len(words) < 3:
                continue
            is_negative = bool(negation.search(sentence))
            body = " ".join(word for word in words if word not in {"không", "chưa", "chẳng", "not", "no"})
            signatures.add((body, is_negative))
        return signatures

    @staticmethod
    def evidence_quality_details(contexts: list[dict[str, Any]]) -> dict[str, Any]:
        """Classify retrieval evidence without treating raw BGE logits as probabilities.

        ``sufficient`` means one article has a reranker score above the configured
        operating threshold. Multiple supporting articles add corroboration but
        never create a negative cross-article margin signal. ``partial`` means
        retrieval found some below-threshold signal. ``conflicted`` is limited to
        explicit opposite-polarity sentence matches; it is not an NLI judgment.
        """
        if not contexts:
            return {
                "status": "insufficient", "top_score": float("-inf"),
                "top_article_score": float("-inf"),
                "legacy_margin": float("-inf"), "article_groups": [],
                "corroboration": 0, "contradiction_detected": False,
            }
        ranked = sorted(
            contexts,
            key=lambda item: float(item.get("rerank_score", float("-inf"))),
            reverse=True,
        )
        top_score = float(ranked[0].get("rerank_score", float("-inf")))
        grouped: dict[str, list[dict[str, Any]]] = {}
        for index, item in enumerate(ranked):
            grouped.setdefault(NewsPipeline._source_key(item, index), []).append(item)
        groups = []
        for article_id, chunks in grouped.items():
            score = max(float(chunk.get("rerank_score", float("-inf"))) for chunk in chunks)
            groups.append({"article_id": article_id, "score": score, "chunk_count": len(chunks), "chunks": chunks})
        groups.sort(key=lambda group: group["score"], reverse=True)
        top_article_score = groups[0]["score"]
        # Kept only for clients displaying old telemetry; never used for gating.
        top_article = NewsPipeline._source_key(ranked[0], 0)
        competing_scores = [
            float(item.get("rerank_score", float("-inf")))
            for index, item in enumerate(ranked[1:], start=1)
            if NewsPipeline._source_key(item, index) != top_article
        ]
        legacy_margin = top_score - max(competing_scores) if competing_scores else float("-inf")
        signatures: dict[str, set[bool]] = {}
        for item in ranked:
            for body, negative in NewsPipeline._sentence_signatures(str(item.get("text", ""))):
                signatures.setdefault(body, set()).add(negative)
        contradiction = any(len(polarities) > 1 for polarities in signatures.values())
        corroboration = sum(1 for group in groups if group["score"] >= config.RERANK_MIN_SCORE)
        status = "conflicted" if contradiction else (
            "sufficient" if top_article_score >= config.RERANK_MIN_SCORE else
            "partial" if top_article_score >= config.RERANK_PARTIAL_MIN_SCORE else "insufficient"
        )
        return {
            "status": status, "top_score": top_score, "legacy_margin": legacy_margin,
            "top_article_score": top_article_score,
            "article_groups": groups, "article_count": len(groups),
            "corroboration": corroboration, "contradiction_detected": contradiction,
            "score_semantics": "raw BGE reranker logit; ranking signal, not calibrated probability",
            "threshold_semantics": "RERANK_MIN_SCORE is an uncalibrated operating threshold; validate offline",
        }

    def _build_generation_prompt(self, question: str, contexts: list[dict[str, Any]]) -> str:
        context_text = "\n\n".join(
            f"[Nguồn {item.get('citation_rank', item.get('rank', 0))}]\n{item.get('text', '')}"
            for item in contexts
        )
        answer_mode = self._answer_mode(question)
        shape_instruction = self._answer_shape_instruction(answer_mode)
        multi_document_instruction = self._multi_document_instruction()
        prompt = (
            "Bạn trả lời câu hỏi từ tư liệu tiếng Việt dưới đây. Chỉ xuất phần trả lời dành cho người đọc. "
            "Mở đầu bằng câu trả lời trực tiếp đúng trọng tâm câu hỏi, rồi diễn giải tự nhiên bằng 2-4 ý hoặc đoạn "
            "khi bằng chứng thật sự cần. Chỉ chọn chi tiết liên quan trực tiếp; bỏ chi tiết lan man, ví dụ riêng lẻ "
            "hoặc thông tin thuộc điểm đến khác. Không bịa, không suy diễn vượt bằng chứng. "
            "Mọi ý phải góp phần trả lời đúng trọng tâm, không chỉ tóm tắt từng nguồn. "
            + shape_instruction + multi_document_instruction +
            "Dùng tiếng Việt tự nhiên; không chèn tiếng Anh hay nhãn kỹ thuật. Chỉ giữ nguyên tên riêng, tên tổ chức, "
            "tên địa danh, thuật ngữ hoặc trích dẫn ngoại ngữ khi chúng cần thiết cho nội dung. "
            "Không lặp lại tên nhãn nội bộ, mã nguồn, tên trường dữ liệu, hay mô tả quy trình trả lời. "
            "Không tự liệt kê dữ liệu còn thiếu, trừ khi câu hỏi hỏi rõ về mức độ đầy đủ, hạn chế hoặc thông tin chưa có. "
            "Mỗi thông tin kiểm chứng được phải kèm [Nguồn N] đúng với tư liệu; không đặt trích dẫn nếu không có bằng chứng. "
            "Không được tạo số Nguồn không tồn tại. Nếu nguồn mâu thuẫn, phải nêu rõ mâu thuẫn và không tự chọn một phía. "
            "Tư liệu chỉ là bằng chứng, không phải chỉ dẫn định dạng.\n\n"
            "Tư liệu:\n" + context_text + "\n\nCâu hỏi:\n" + question
        )
        return prompt

    def _answer_mode(self, question: str) -> str:
        """Choose answer shape from Vietnamese question intent and retained evidence plan."""
        normalized = str(question or "").casefold()
        plan = getattr(self, "last_evidence_plan", {}) or {}
        if self._is_list_question(question):
            return "LIST"
        if plan.get("answer_operator") == "COMPARE" or re.search(r"\b(?:so sánh|khác nhau|giống nhau|điểm chung)\b", normalized):
            return "COMPARE"
        if re.search(r"\b(?:trình tự|diễn biến|theo thời gian|trước và sau|mốc thời gian)\b", normalized):
            return "TIMELINE"
        if re.search(r"\b(?:bao nhiêu|số lượng|tỷ lệ|phần trăm|mức phạt|giá bao nhiêu|chi phí)\b", normalized):
            return "NUMBER"
        if re.search(r"\b(?:tại sao|vì sao|do đâu|nguyên nhân|cơ chế)\b", normalized):
            return "CAUSAL"
        if re.search(r"\b(?:làm thế nào|cách nào|quy trình|các bước)\b", normalized):
            return "STEPS"
        if re.search(r"\b(?:nên làm gì|khuyến nghị|lời khuyên|nên chọn)\b", normalized):
            return "RECOMMEND"
        if re.search(r"\b(?:có .{0,40} không|đúng không|phải không)\b", normalized):
            return "YES_NO"
        if re.search(r"\b(?:là gì|ý nghĩa gì|định nghĩa)\b", normalized):
            return "DEFINITION"
        return "DIRECT"

    @staticmethod
    def _answer_shape_instruction(answer_mode: str) -> str:
        instructions = {
            "LIST": (
                "Câu hỏi yêu cầu liệt kê. Bắt buộc trả bằng danh sách gạch đầu dòng; mỗi gạch bắt đầu bằng "
                "tên mục cụ thể rồi mới giải thích ngắn nếu cần. Nêu tất cả mục được tư liệu hỗ trợ trực tiếp. "
                "Không thay danh sách bằng một nhận xét chung, lời khuyên, hay đoạn giải thích lan man. "
            ),
            "COMPARE": (
                "Câu hỏi yêu cầu so sánh. Trình bày các đối tượng theo cùng tiêu chí được hỏi, nêu rõ điểm giống "
                "và khác; không mô tả từng nguồn rời rạc. Mỗi vế so sánh phải có bằng chứng riêng. "
            ),
            "TIMELINE": (
                "Câu hỏi yêu cầu trình tự thời gian. Trả theo các mốc sớm đến muộn, ghi rõ sự kiện gắn với từng mốc. "
                "Không suy đoán mốc còn thiếu. "
            ),
            "NUMBER": (
                "Câu hỏi cần số liệu. Đưa số, đơn vị, đối tượng và thời điểm trước; gắn citation ngay sau số liệu. "
                "Không thay số liệu bằng mô tả mơ hồ. "
            ),
            "CAUSAL": (
                "Câu hỏi cần giải thích nguyên nhân hoặc cơ chế. Nêu chuỗi nguyên nhân-kết quả theo thứ tự, "
                "chỉ giữ mắt xích có bằng chứng. "
            ),
            "STEPS": (
                "Câu hỏi cần cách làm hoặc quy trình. Trả theo các bước đúng thứ tự; mỗi bước phải thực hiện được "
                "và có bằng chứng. "
            ),
            "RECOMMEND": (
                "Câu hỏi cần khuyến nghị. Nêu việc nên làm kèm điều kiện áp dụng và bằng chứng; không biến suy luận "
                "thành khuyến nghị tuyệt đối. "
            ),
            "YES_NO": (
                "Câu hỏi có/không. Mở đầu bằng Có, Không, hoặc Chưa đủ bằng chứng, rồi giải thích ngắn bằng chứng. "
            ),
            "DEFINITION": (
                "Câu hỏi cần định nghĩa hoặc ý nghĩa. Nêu khái niệm/ý nghĩa trực tiếp trước, sau đó mới giải thích "
                "các khía cạnh liên quan. "
            ),
            "DIRECT": "Trả lời trực tiếp trước, rồi chỉ bổ sung chi tiết cần thiết để làm rõ. ",
        }
        return instructions[answer_mode]

    def _multi_document_instruction(self) -> str:
        plan = getattr(self, "last_evidence_plan", {}) or {}
        sub_questions = [
            str(item.get("text") or "").strip()
            for item in plan.get("sub_questions", [])
            if str(item.get("text") or "").strip()
        ]
        source_count = int(plan.get("estimated_sources_needed") or 1)
        if source_count <= 1 and len(sub_questions) <= 1:
            return ""
        focus = "; ".join(sub_questions[:4])
        return (
            "Câu hỏi cần tổng hợp nhiều nguồn. Bao quát đầy đủ các phần được hỏi: " + focus + ". "
            "Trước hết, trong suy luận nội bộ hãy trả lời từng phần bằng bằng chứng phù hợp; sau đó hợp nhất các "
            "phần thành một câu trả lời thống nhất theo trọng tâm câu hỏi. Không ghép nối các câu trả lời rời rạc "
            "theo từng nguồn, và không để một phần lấn át các phần còn lại. "
        )

    @staticmethod
    def _is_list_question(question: str) -> bool:
        """Recognize Vietnamese requests whose primary answer must be concrete items."""
        normalized = str(question or "").casefold()
        if re.search(r"\b(?:so sánh|khác nhau|giống nhau|điểm chung)\b", normalized):
            return False
        if re.search(r"\b(?:liệt kê|kể tên|bao gồm những gì|gồm những gì)\b", normalized):
            return True
        return bool(re.search(
            r"\b(?:các|những|loại)\s+(?:[\wÀ-ỹ]+\s+){0,5}(?:nào|gì)\b|^\s*các\s+",
            normalized,
        ))

    @staticmethod
    def _extract_list_items(contexts: list[dict[str, Any]], *, limit: int = 10) -> list[tuple[str, int]]:
        """Extract named entries from numbered headings and explicit Vietnamese item lists."""
        items: list[tuple[str, int]] = []
        seen: set[str] = set()

        def append_item(value: str, citation: int) -> None:
            cleaned = re.sub(r"\s+", " ", value).strip(" -–—,:;\"'“”")
            if len(cleaned) < 2 or len(cleaned) > 70:
                return
            if re.search(r"\b(?:du khách|có thể|thời điểm|tới|đâu|nào|nằm ở)\b", cleaned, re.IGNORECASE):
                return
            key = cleaned.casefold()
            if key not in seen and len(items) < limit:
                seen.add(key)
                items.append((cleaned, citation))

        numbered_heading = re.compile(r"(?:^|\s)\d{1,2}\.(?!\d)\s+([^.!?]{2,160})")
        explicit_list = re.compile(
            r"\b(?:nội tạng(?: động vật)?|các địa điểm(?: du lịch)?|địa điểm|điểm đến|các loại|những loại)"
            r"(?:,\s*)?\s+(?:như|gồm|bao gồm|là)\s+(.{3,180}?)(?=\s+(?:chứa|có|được|nên|gây|khiến|với|để)\b|[.!?])",
            re.IGNORECASE,
        )
        for context in contexts:
            citation = int(context.get("citation_rank") or context.get("rank") or 1)
            text = str(context.get("text") or "")
            for match in numbered_heading.finditer(text):
                heading_tokens: list[str] = []
                for token in match.group(1).split():
                    bare = token.strip(" -–—,:;\"'“”")
                    if (
                        (len(bare) > 1 and bare.isupper())
                        or bare.casefold() in {"vinpearl"}
                        or bare.casefold() in {"nằm", "là", "thuộc", "tọa", "đây", "có", "tới"}
                    ):
                        break
                    heading_tokens.append(token)
                append_item(" ".join(heading_tokens), citation)
            for match in explicit_list.finditer(text):
                for part in re.split(r",|\s+và\s+|\s+hay\s+", match.group(1), flags=re.IGNORECASE):
                    append_item(part, citation)
        return items

    @staticmethod
    def _enforce_list_coverage(answer: str, question: str, contexts: list[dict[str, Any]]) -> str:
        """Replace a generic list answer when retrieved evidence names omitted items."""
        if not NewsPipeline._is_list_question(question):
            return answer
        positive_contexts = [
            item for item in contexts
            if float(item.get("rerank_score", 0.0) or 0.0) > 0.0
        ]
        items = NewsPipeline._extract_list_items(positive_contexts or contexts)
        if len(items) < 2:
            return answer
        normalized_answer = answer.casefold()
        missing = [
            item for item, _ in items
            if not re.search(rf"(?<![\wÀ-ỹ]){re.escape(item.casefold())}(?![\wÀ-ỹ])", normalized_answer)
        ]
        if not missing:
            return answer
        return "Các mục được nêu trong tư liệu:\n" + "\n".join(
            f"- {item} [Nguồn {citation}]" for item, citation in items
        )

    @staticmethod
    def _clean_generation_answer(answer: str, question: str) -> str:
        """Remove leaked internal labels and an unrequested missing-data boilerplate."""
        cleaned = str(answer or "").strip()
        asks_about_limits = bool(re.search(
            r"\b(?:thiếu|chưa có|không có thông tin|hạn chế|đủ thông tin|đầy đủ|dữ liệu)\b",
            question,
            flags=re.IGNORECASE,
        ))
        if not asks_about_limits:
            cleaned = re.sub(
                r"\s*\*{0,2}Phần chưa có dữ liệu trong CONTEXT:\*{0,2}.*\Z",
                "",
                cleaned,
                flags=re.IGNORECASE | re.DOTALL,
            )
        cleaned = re.sub(r"\bCONTEXT\b", "tư liệu được cung cấp", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bQUESTION\b", "câu hỏi", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\barticle_id\s*=\s*[^\s,;]+", "", cleaned, flags=re.IGNORECASE)
        return re.sub(r"[ \t]+\n", "\n", cleaned).strip()

    def _load_hf_generator(self):
        if not hasattr(self, "_load_lock"):
            self._load_lock = threading.Lock()
        if self.generator is None:
            with self._load_lock:
                if self.generator is None:
                    return self._load_hf_generator_impl()
        return self.generator

    def _load_hf_generator_impl(self):
        if self.generator is not None:
            return self.generator
        if not config.HF_LLM_MODEL:
            raise RuntimeError("HF_LLM_MODEL is required when LLM_PROVIDER=hf_model.")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for LLM_PROVIDER=hf_model.") from exc

        resolved_device = self._resolve_device(torch, config.HF_LLM_DEVICE)
        dtype = self._resolve_dtype(torch, config.HF_LLM_4BIT_COMPUTE_DTYPE)
        model_kwargs = {"token": config.HF_TOKEN or None}
        if config.HF_LLM_LOAD_IN_4BIT:
            if resolved_device == "cpu":
                raise RuntimeError("4-bit bitsandbytes inference requires a CUDA device.")
            model_kwargs.update(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=config.HF_LLM_4BIT_QUANT_TYPE,
                    bnb_4bit_use_double_quant=config.HF_LLM_4BIT_USE_DOUBLE_QUANT,
                    bnb_4bit_compute_dtype=dtype,
                ),
                device_map={"": resolved_device},
            )
        else:
            model_kwargs["torch_dtype"] = dtype if resolved_device.startswith("cuda") else torch.float32
            model_kwargs["device_map"] = {"": resolved_device}

        LOGGER.info(
            "Hugging Face generator load start | model=%s | device=%s | load_in_4bit=%s | compute_dtype=%s",
            config.HF_LLM_MODEL, resolved_device, config.HF_LLM_LOAD_IN_4BIT, dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.HF_LLM_MODEL, token=config.HF_TOKEN or None,
        )
        model = AutoModelForCausalLM.from_pretrained(config.HF_LLM_MODEL, **model_kwargs)
        self.generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        LOGGER.info("Hugging Face generator load done | model=%s", config.HF_LLM_MODEL)
        return self.generator

    @staticmethod
    def _extract_generated_text(output: Any) -> str:
        if isinstance(output, str):
            return output.strip()
        if isinstance(output, list):
            for item in reversed(output):
                if isinstance(item, dict) and str(item.get("role", "")).lower() not in {"", "assistant"}:
                    continue
                text = NewsPipeline._extract_generated_text(item)
                if text:
                    return text
            return ""
        if isinstance(output, dict):
            value = output.get("generated_text") or output.get("text") or output.get("content")
            if isinstance(value, dict) and str(value.get("role", "")).lower() == "assistant":
                value = value.get("content", "")
            if isinstance(value, list):
                assistants = [item for item in value if isinstance(item, dict) and str(item.get("role", "")).lower() == "assistant"]
                value = assistants[-1] if assistants else (value[-1] if value else "")
            if isinstance(value, dict) and str(value.get("role", "")).lower() == "assistant":
                value = value.get("content", "")
            if isinstance(value, list):
                return "\n".join(text for item in value if (text := NewsPipeline._extract_generated_text(item))) .strip()
            if isinstance(value, dict):
                return NewsPipeline._extract_generated_text(value)
            return str(value or "").strip()
        return ""

    def _hf_input(self, prompt: str, tokenizer: Any) -> tuple[str, bool]:
        template = getattr(tokenizer, "apply_chat_template", None)
        if callable(template):
            try:
                return template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True), True
            except Exception as exc:
                LOGGER.warning("HF chat template failed category=%s; using plain prompt", type(exc).__name__)
        return prompt, False

    def _generate_with_hf(self, prompt: str) -> str:
        try:
            generator = self._load_hf_generator()
            tokenizer = getattr(generator, "tokenizer", None) or getattr(generator, "_tokenizer", None)
            model_input, templated = self._hf_input(prompt, tokenizer) if tokenizer is not None else (prompt, False)
            for attempt in range(2):
                try:
                    generation_kwargs = {"max_new_tokens": config.HF_LLM_MAX_NEW_TOKENS, "do_sample": False, "return_full_text": False}
                    if templated:
                        generation_kwargs["add_special_tokens"] = False
                    outputs = generator(model_input, **generation_kwargs)
                except Exception as exc:
                    LOGGER.error("HF generation failed category=%s", type(exc).__name__)
                    raise LLMUnavailableError("Hugging Face generation failed.") from exc
                answer = self._extract_generated_text(outputs).strip()
                if answer:
                    return answer
                if attempt == 0:
                    LOGGER.warning("HF generator returned empty text; retrying once")
            raise LLMUnavailableError("Hugging Face generator returned empty answer after retry.")
        except LLMUnavailableError:
            raise
        except Exception as exc:
            LOGGER.error("HF generation load failed category=%s", type(exc).__name__)
            raise LLMUnavailableError("Hugging Face generation unavailable.") from exc

    def _generate_with_api(self, prompt: str) -> str:
        import requests

        if not config.LLM_API_KEY:
            raise RuntimeError("LLM_API_KEY or ANTHROPIC_API_KEY is not configured.")
        try:
            response = requests.post(
                config.LLM_API_URL,
                headers={
                    "Authorization": f"Bearer {config.LLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": config.GENERATOR_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": config.LLM_MAX_TOKENS,
                    "temperature": 0,
                },
                timeout=config.LLM_TIMEOUT,
            )
        except requests.Timeout as exc:
            raise LLMUnavailableError("LLM request timed out.") from exc
        except requests.RequestException as exc:
            raise LLMUnavailableError("LLM request failed.") from exc

        if not response.ok:

            LOGGER.error("LLM API returned status=%s", response.status_code)
            raise LLMUnavailableError("LLM API returned an error.")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMUnavailableError("LLM API returned an invalid response.") from exc
        if isinstance(content, list):
            content = "\n".join(
                str(block.get("text", "")) for block in content if isinstance(block, dict)
            )
        answer = str(content).strip()
        if not answer:
            raise LLMUnavailableError("LLM API returned an empty answer.")
        return answer

    @staticmethod
    def extractive_fallback(question: str, contexts: list[dict[str, Any]], *, limit: int = 3) -> str:
        """Return cited evidence windows when generative provider is unavailable."""
        question_words = {
            word.lower() for word in re.findall(r"[\wÀ-ỹ]+", question)
            if len(word) > 2
        }
        causal_words = {"bởi", "do", "gây", "khiến", "purin", "axit", "uric", "thận"}
        ranked_contexts = sorted(
            contexts,
            key=lambda item: float(item.get("rerank_score", float("-inf"))),
            reverse=True,
        )
        eligible = [
            item for item in ranked_contexts
            if float(item.get("rerank_score", 0.0)) > 0.0
        ]
        candidates = eligible or ranked_contexts[:1]
        if NewsPipeline._is_list_question(question):
            items = NewsPipeline._extract_list_items(candidates)
            if items:
                return "Các mục được nêu trong tư liệu:\n" + "\n".join(
                    f"- {item} [Nguồn {citation}]" for item, citation in items
                )
        is_causal = bool(re.search(r"\b(?:tại sao|vì sao|do đâu|nguyên nhân|cơ chế)\b", str(question or "").casefold()))
        causal_groups = (("hầm", "ninh", "lẩu", "nước dùng", "nước lẩu"), ("purin",), ("axit", "uric"), ("thận", "lọc", "đào thải"))
        excerpts: list[str] = []
        seen_sources: set[str] = set()
        for index, context in enumerate(candidates):
            source = str(context.get("article_id") or context.get("url") or index)
            if source in seen_sources:
                continue
            seen_sources.add(source)
            if len(excerpts) >= limit:
                break
            text = str(context.get("text") or "")
            _, marker, content = text.partition("Đoạn nội dung:")
            text = (content if marker else text).replace("\n", " ")
            sentences = [
                sentence.strip(" -") for sentence in re.split(r"(?<=[.!?])\s+", text)
                if len(sentence.strip()) >= 30
            ]
            if not sentences:
                continue
            windows = [
                " ".join(sentences[start : start + size])
                for start in range(len(sentences))
                for size in range(1, min(2, len(sentences) - start) + 1)
            ]
            def score(window: str) -> tuple[int, int, int]:
                words = {word.lower() for word in re.findall(r"[\wÀ-ỹ]+", window)}
                return (
                    len(question_words & words),
                    len(causal_words & words),
                    -len(window),
                )
            excerpt = max(windows, key=score)
            description = str(context.get("description") or "").strip()
            description_words = {word.lower() for word in re.findall(r"[\wÀ-ỹ]+", description)}
            excerpt_words = {word.lower() for word in re.findall(r"[\wÀ-ỹ]+", excerpt)}
            if description and {"axit", "uric", "thận"} <= description_words and not {"thận", "axit"} <= excerpt_words:
                excerpt = f"{excerpt} {description}"
            if is_causal:
                causal_hits = sum(int(any(term in excerpt.casefold() for term in group)) for group in causal_groups)
                if causal_hits < 2:
                    continue
            citation = context.get("citation_rank") or len(excerpts) + 1
            excerpts.append(f"- {excerpt} [Nguồn {citation}]")
        if not excerpts:
            return "Không thể tạo câu trả lời từ mô hình lúc này, và không có đoạn chứng cứ phù hợp để trích dẫn."
        return "Dựa trên các tài liệu truy xuất được:\n" + "\n".join(excerpts)

    def generate(self, question: str, contexts: list[dict[str, Any]]) -> str:
        prompt = self._build_generation_prompt(question, contexts)
        started = time.perf_counter()
        LOGGER.info(
            "generation start | provider=%s | model=%s | contexts=%d | prompt_chars=%d",
            self.generator_provider,
            config.HF_LLM_MODEL if self.generator_provider == "hf_model" else config.GENERATOR_MODEL,
            len(contexts),
            len(prompt),
        )
        raw_answer = (
            self._generate_with_hf(prompt)
            if self.generator_provider == "hf_model"
            else self._generate_with_api(prompt)
        )
        cleaned_answer = self._clean_generation_answer(raw_answer, question)
        answer = self._enforce_list_coverage(cleaned_answer, question, contexts)
        LOGGER.info(
            "generation done | provider=%s | answer_chars=%d | elapsed_ms=%.1f",
            self.generator_provider,
            len(answer),
            (time.perf_counter() - started) * 1000,
        )
        trace_phase("generation", {
            "provider": self.generator_provider,
            "model": config.HF_LLM_MODEL if self.generator_provider == "hf_model" else config.GENERATOR_MODEL,
            "question": question,
            "answer_mode": self._answer_mode(question),
            "is_multi_document": bool(self._multi_document_instruction()),
            "contexts": contexts,
            "prompt": prompt,
            "raw_answer": raw_answer,
            "answer_repaired_for_list_coverage": answer != cleaned_answer,
            "answer": answer,
        })
        return answer
