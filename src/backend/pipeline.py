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
from .source_identity import source_key

LOGGER = logging.getLogger("rag-api.pipeline")

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
            LOGGER.info(
                "reranker model load done | device=%s | dtype=%s | elapsed_ms=%.1f",
                device, dtype, (time.perf_counter() - started) * 1000,
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
            "rerank done | model=%s | candidates=%d | top_score=%.4f | elapsed_ms=%.1f",
            config.RERANKER_MODEL, len(results), float(results[0].get("rerank_score", 0.0)) if results else 0.0,
            (time.perf_counter() - started) * 1000,
        )
        return results

    def search(self, question: str, top_k: int = config.TOP_K_CONTEXT) -> list[dict[str, Any]]:
        contexts, _, _, _ = self.search_with_evidence(question, top_k)
        return contexts

    def search_with_evidence(
        self,
        question: str,
        top_k: int = config.TOP_K_CONTEXT,
    ) -> tuple[list[dict[str, Any]], bool, float, float]:
        """Evaluate evidence on the full reranked pool, then select final contexts."""
        ranked = self.rerank(question, self.retrieve(question))
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
        return selected, sufficient, top_score, margin

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
            "article_groups": groups, "article_count": len(groups),
            "corroboration": corroboration, "contradiction_detected": contradiction,
            "score_semantics": "raw BGE reranker logit; ranking signal, not calibrated probability",
            "threshold_semantics": "RERANK_MIN_SCORE is an uncalibrated operating threshold; validate offline",
        }

    def _build_generation_prompt(self, question: str, contexts: list[dict[str, Any]]) -> str:
        context_text = "\n\n".join(
            f"[Nguồn {item.get('citation_rank', item.get('rank', 0))}] article_id={item.get('article_id')} "
            f"title={item.get('title')}\n{item.get('text', '')}"
            for item in contexts
        )
        prompt = (
            "Bạn là hệ thống hỏi đáp RAG cho tin tức tiếng Việt. "
            "Hãy trả lời đầy đủ và có chiều sâu, không trả lời cụt ngủn. "
            "Chỉ sử dụng thông tin có trong CONTEXT; không được bịa hoặc suy diễn vượt quá bằng chứng. "
            "Hãy tổng hợp các context liên quan, nêu rõ nguyên nhân, diễn biến, tác động hoặc khuyến nghị "
            "nếu những thông tin đó có trong context. Ưu tiên các chi tiết cụ thể. "
            "Trình bày khoảng 3-6 đoạn hoặc danh sách 5-10 ý tùy câu hỏi. "
            "Nếu context không đủ bằng chứng, phải nói rõ phần nào chưa có dữ liệu.\n\n"
            "CONTEXT:\n" + context_text + "\n\nQUESTION:\n" + question + "\n\n"
            "Mỗi claim có thể kiểm chứng phải gắn đúng citation [Nguồn N] theo CONTEXT; không gắn citation nếu không có bằng chứng. Trả lời bằng tiếng Việt."
        )
        return prompt

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
        answer = (
            self._generate_with_hf(prompt)
            if self.generator_provider == "hf_model"
            else self._generate_with_api(prompt)
        )
        LOGGER.info(
            "generation done | provider=%s | answer_chars=%d | elapsed_ms=%.1f",
            self.generator_provider,
            len(answer),
            (time.perf_counter() - started) * 1000,
        )
        return answer
