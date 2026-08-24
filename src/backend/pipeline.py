from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from . import config

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
    """E5-large retrieval -> BGE reranking -> configurable LLM generation."""

    def __init__(self) -> None:
        self.client = QdrantClient(path=str(config.QDRANT_PATH))
        self.encoder = None
        self.reranker = None
        self.generator = None
        self.generator_provider = config.resolve_llm_provider()

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

    def _load_encoder(self):
        if self.encoder is None:
            started = time.perf_counter()
            LOGGER.info("embedding model load start | model=%s", config.EMBEDDING_MODEL)
            import torch
            from sentence_transformers import SentenceTransformer

            device = self._resolve_device(torch)
            self.encoder = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
            LOGGER.info("embedding model load done | elapsed_ms=%.1f", (time.perf_counter() - started) * 1000)
        return self.encoder

    def _load_reranker(self):
        if self.reranker is None:
            started = time.perf_counter()
            LOGGER.info("reranker model load start | model=%s", config.RERANKER_MODEL)
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device = self._resolve_device(torch)
            tokenizer = AutoTokenizer.from_pretrained(config.RERANKER_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(config.RERANKER_MODEL)
            model.to(device)
            model.eval()
            self.reranker = (tokenizer, model, torch, device)
            LOGGER.info("reranker model load done | device=%s | elapsed_ms=%.1f", device, (time.perf_counter() - started) * 1000)
        return self.reranker

    def is_ready(self) -> bool:
        try:
            return self.client.collection_exists(config.COLLECTION)
        except Exception:
            return False

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
        encoder = self._load_encoder()
        vectors = encoder.encode(
            ["passage: " + str(row["text"]).strip() for row in rows],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        vector_size = int(vectors.shape[1])
        if self.is_ready():
            self.client.delete_collection(config.COLLECTION)
        self.client.create_collection(
            collection_name=config.COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        points = []
        for index, (row, vector) in enumerate(zip(rows, vectors)):
            metadata = row.get("metadata") or {}
            payload = {
                "chunk_id": str(row.get("chunk_id", index)),
                "article_id": str(row.get("article_id", metadata.get("article_id", ""))),
                "title": metadata.get("title", ""),
                "description": metadata.get("description", ""),
                "category": metadata.get("category", ""),
                "url": metadata.get("url", ""),
                "chunk_index": metadata.get("chunk_index", row.get("chunk_index", 0)),
                "text": str(row.get("text", "")),
            }
            points.append(PointStruct(id=index, vector=vector.tolist(), payload=payload))
            if len(points) >= batch_size:
                self.client.upsert(collection_name=config.COLLECTION, points=points)
                points = []
        if points:
            self.client.upsert(collection_name=config.COLLECTION, points=points)
        return len(rows)

    def retrieve(self, question: str, limit: int = config.TOP_K_RETRIEVAL) -> list[dict[str, Any]]:
        started = time.perf_counter()
        if not self.is_ready():
            raise IndexUnavailableError("Qdrant index is not ready. Run build_index.py first.")
        vector = self._load_encoder().encode(
            ["query: " + question.strip()], normalize_embeddings=True
        )[0].tolist()
        try:
            response = self.client.query_points(
                collection_name=config.COLLECTION,
                query=vector,
                limit=limit,
                with_payload=True,
            )
        except Exception as exc:
            raise IndexUnavailableError("Qdrant query failed.") from exc
        results = [
            {**(hit.payload or {}), "retrieval_score": float(hit.score)}
            for hit in response.points
        ]
        LOGGER.info(
            "retrieval done | requested=%d | returned=%d | top_score=%.4f | elapsed_ms=%.1f",
            limit, len(results), float(results[0].get("retrieval_score", 0.0)) if results else 0.0,
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
        sufficient, top_score, margin = self.evidence_quality(ranked)
        return ranked[:top_k], sufficient, top_score, margin

    @staticmethod
    def evidence_quality(contexts: list[dict[str, Any]]) -> tuple[bool, float, float]:
        """Return whether BGE evidence is strong enough to send to the LLM."""
        if not contexts:
            return False, float("-inf"), float("-inf")
        ranked = sorted(
            contexts,
            key=lambda item: float(item.get("rerank_score", float("-inf"))),
            reverse=True,
        )
        top_score = float(ranked[0].get("rerank_score", float("-inf")))
        top_article = str(ranked[0].get("article_id", ""))
        competing_scores = [
            float(item.get("rerank_score", float("-inf")))
            for item in ranked[1:]
            if str(item.get("article_id", "")) != top_article
        ]
        # A single result/article cannot establish a meaningful separation margin.
        # Treat that case conservatively so top_k=1 never bypasses the gate.
        margin = top_score - max(competing_scores) if competing_scores else float("-inf")
        sufficient = top_score >= config.RERANK_MIN_SCORE and margin >= config.RERANK_MIN_MARGIN
        return sufficient, top_score, margin

    def _build_generation_prompt(self, question: str, contexts: list[dict[str, Any]]) -> str:
        context_text = "\n\n".join(
            f"[Nguồn {item['rank']}] article_id={item.get('article_id')} "
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
            "Trả lời bằng tiếng Việt."
        )
        return prompt

    def _load_hf_generator(self):
        if self.generator is not None:
            return self.generator
        if not config.HF_LLM_MODEL:
            raise RuntimeError("HF_LLM_MODEL is required when LLM_PROVIDER=hf_model.")
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for LLM_PROVIDER=hf_model.") from exc

        resolved_device = self._resolve_device(torch, config.HF_LLM_DEVICE)
        pipeline_device = -1 if resolved_device == "cpu" else int(resolved_device.split(":", 1)[1])

        LOGGER.info("Hugging Face generator load start | model=%s | device=%s", config.HF_LLM_MODEL, resolved_device)
        self.generator = pipeline(
            "text-generation",
            model=config.HF_LLM_MODEL,
            tokenizer=config.HF_LLM_MODEL,
            token=config.HF_TOKEN or None,
            device=pipeline_device,
        )
        LOGGER.info("Hugging Face generator load done | model=%s", config.HF_LLM_MODEL)
        return self.generator

    def _generate_with_hf(self, prompt: str) -> str:
        generator = self._load_hf_generator()
        outputs = generator(
            prompt,
            max_new_tokens=config.HF_LLM_MAX_NEW_TOKENS,
            do_sample=False,
            return_full_text=False,
        )
        if not outputs or not isinstance(outputs[0], dict):
            raise RuntimeError("Hugging Face generator returned an invalid response")
        answer = str(outputs[0].get("generated_text", "")).strip()
        if not answer:
            raise RuntimeError("Hugging Face generator returned an empty answer")
        return answer

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
            detail = response.text[:500].replace("\n", " ")
            LOGGER.error("LLM API returned status=%s detail=%s", response.status_code, detail)
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
            raise RuntimeError("LLM API returned an empty answer")
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
