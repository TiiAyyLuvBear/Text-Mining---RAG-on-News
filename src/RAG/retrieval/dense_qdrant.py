from __future__ import annotations

import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.RAG.embed.embed_chunks import DEFAULT_MODEL, load_sentence_transformer, prepare_document_text, prepare_query_text

from .schema import read_jsonl


@lru_cache(maxsize=None)
def _encoder(model_name: str):
    """Keep E5 loaded on the selected device for all queries in this process."""
    return load_sentence_transformer(model_name)


def _client(path: str | Path):
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("qdrant-client is required for dense retrieval.") from exc
    return QdrantClient(path=str(path))


def build_qdrant_index(chunks_path: str | Path, qdrant_path: str | Path, collection: str, *, model_name: str = DEFAULT_MODEL, batch_size: int = 16, rebuild: bool = False) -> dict[str, object]:
    from qdrant_client import models

    rows = [row for row in read_jsonl(chunks_path) if str(row.get("text") or "").strip()]
    encoder = _encoder(model_name)
    vectors = np.asarray(encoder.encode([prepare_document_text(str(row["text"])) for row in rows], batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)
    client = _client(qdrant_path)
    if client.collection_exists(collection):
        if not rebuild:
            raise ValueError(f"Collection '{collection}' already exists; use --rebuild to replace it.")
        client.delete_collection(collection)
    client.create_collection(collection, vectors_config=models.VectorParams(size=int(vectors.shape[1]), distance=models.Distance.COSINE))
    points = [models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, str(row["chunk_id"]))), vector=vector.tolist(), payload={"chunk_id": str(row["chunk_id"]), "article_id": str(row["article_id"]), "text": str(row.get("text") or ""), "chunk_text": str(row.get("chunk_text") or ""), **dict(row.get("metadata") or {})}) for row, vector in zip(rows, vectors)]
    client.upsert(collection, points=points, wait=True)
    return {"qdrant_path": str(qdrant_path), "collection": collection, "chunks": len(rows), "model": model_name, "dimension": int(vectors.shape[1])}


def search_qdrant(qdrant_path: str | Path, collection: str, query: str, *, top_k: int, model_name: str = DEFAULT_MODEL) -> list[dict[str, object]]:
    encoder = _encoder(model_name)
    vector = np.asarray(encoder.encode([prepare_query_text(query)], normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)[0].tolist()
    client = _client(qdrant_path)
    response = client.query_points(collection_name=collection, query=vector, limit=top_k, with_payload=True)
    points = response.points
    return [{"chunk_id": point.payload["chunk_id"], "article_id": point.payload["article_id"], "score": float(point.score), "text": point.payload.get("text", ""), "chunk_text": point.payload.get("chunk_text", ""), "title": point.payload.get("title"), "category": point.payload.get("category"), "chunk_index": point.payload.get("chunk_index")} for point in points]
