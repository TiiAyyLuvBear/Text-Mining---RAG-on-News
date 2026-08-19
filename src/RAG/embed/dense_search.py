from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .embed_chunks import DEFAULT_MODEL, Encoder, load_sentence_transformer, prepare_query_text
except ImportError:  # Allows `python src\embed\dense_search.py ...`
    from embed_chunks import DEFAULT_MODEL, Encoder, load_sentence_transformer, prepare_query_text


def read_metadata(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_index(index_dir: str | Path) -> tuple[np.ndarray, list[dict[str, object]]]:
    base = Path(index_dir)
    embeddings = np.load(base / "embeddings.npy")
    metadata = read_metadata(base / "metadata.jsonl")
    if len(embeddings) != len(metadata):
        raise ValueError(f"Index mismatch: {len(embeddings)} vectors but {len(metadata)} metadata rows")
    return embeddings, metadata


def search_dense_index(
    *,
    query: str,
    embeddings: np.ndarray,
    metadata: list[dict[str, object]],
    encoder: Encoder,
    top_k: int,
) -> list[dict[str, object]]:
    query_vector = encoder.encode([prepare_query_text(query)], normalize_embeddings=True, show_progress_bar=False)
    query_array = np.asarray(query_vector, dtype=np.float32)[0]
    scores = embeddings @ query_array
    limit = min(top_k, len(scores))
    top_indices = np.argsort(scores)[::-1][:limit]
    results: list[dict[str, object]] = []
    for index in top_indices:
        row = metadata[int(index)]
        results.append(
            {
                "chunk_id": row.get("chunk_id"),
                "article_id": row.get("article_id"),
                "score": float(scores[int(index)]),
                "title": row.get("title"),
                "category": row.get("category"),
                "strategy": row.get("strategy"),
                "chunk_index": row.get("chunk_index"),
                "text": row.get("text"),
                "chunk_text": row.get("chunk_text"),
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search a local dense embedding index.")
    parser.add_argument("--index-dir", required=True, help="Directory containing embeddings.npy and metadata.jsonl.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    embeddings, metadata = load_index(args.index_dir)
    encoder = load_sentence_transformer(args.model)
    results = search_dense_index(query=args.query, embeddings=embeddings, metadata=metadata, encoder=encoder, top_k=args.top_k)
    print(json.dumps({"query": args.query, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



