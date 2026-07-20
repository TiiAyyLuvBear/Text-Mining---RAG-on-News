from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from . import config


def _load_chunks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class NewsPipeline:
    """E5-large retrieval -> Jina token reranking -> Claude generation."""

    def __init__(self) -> None:
        self.client = QdrantClient(path=str(config.QDRANT_PATH))
        self.encoder = None
        self.reranker = None
        self.anthropic = None

    def _load_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(config.EMBEDDING_MODEL)
        return self.encoder

    def _load_reranker(self):
        if self.reranker is None:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(config.RERANKER_MODEL, trust_remote_code=True)
        return self.reranker

    def is_ready(self) -> bool:
        try:
            return self.client.collection_exists(config.COLLECTION)
        except Exception:
            return False

    def build_index(self, chunk_path: Path | None = None, batch_size: int = 64) -> int:
        path = chunk_path or config.CHUNK_PATH
        rows = _load_chunks(path)
        rows = [row for row in rows if str(row.get("text") or "").strip()]
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
        if not self.is_ready():
            raise RuntimeError("Qdrant index is not ready. Run build_index.py first.")
        vector = self._load_encoder().encode(
            ["query: " + question.strip()], normalize_embeddings=True
        )[0].tolist()
        response = self.client.query_points(
            collection_name=config.COLLECTION,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return [
            {**(hit.payload or {}), "retrieval_score": float(hit.score)}
            for hit in response.points
        ]

    def rerank(self, question: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        try:
            scores = self._load_reranker().predict(
                [(question, str(candidate.get("text", ""))) for candidate in candidates]
            )
            ranked = [
                {**candidate, "rerank_score": float(score)}
                for candidate, score in zip(candidates, scores)
            ]
        except Exception:
            # Keep the API usable when the Jina model is unavailable locally.
            ranked = [
                {**candidate, "rerank_score": candidate.get("retrieval_score", 0.0)}
                for candidate in candidates
            ]
        ranked.sort(key=lambda item: float(item["rerank_score"]), reverse=True)
        return [{**item, "rank": index + 1} for index, item in enumerate(ranked)]

    def search(self, question: str, top_k: int = config.TOP_K_CONTEXT) -> list[dict[str, Any]]:
        return self.rerank(question, self.retrieve(question))[:top_k]

    def generate(self, question: str, contexts: list[dict[str, Any]]) -> str:
        import os
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured.")
        kwargs = {"api_key": api_key}
        if config.ANTHROPIC_BASE_URL:
            kwargs["base_url"] = config.ANTHROPIC_BASE_URL
        self.anthropic = self.anthropic or Anthropic(**kwargs)
        context_text = "\n\n".join(
            f"[Nguồn {item['rank']}] article_id={item.get('article_id')} "
            f"title={item.get('title')}\n{item.get('text', '')}"
            for item in contexts
        )
        prompt = (
            "Bạn là hệ thống hỏi đáp tin tức tiếng Việt.\n"
            "Chỉ dùng CONTEXT để trả lời QUESTION. Không suy đoán.\n"
            "Nếu không đủ bằng chứng, trả lời: Không đủ thông tin trong dữ liệu được cung cấp.\n"
            "Trả lời ngắn gọn và thêm mục Nguồn gồm article_id, tiêu đề.\n\n"
            f"QUESTION:\n{question}\n\nCONTEXT:\n{context_text}"
        )
        response = self.anthropic.messages.create(
            model=config.GENERATOR_MODEL,
            max_tokens=700,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "\n".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()
