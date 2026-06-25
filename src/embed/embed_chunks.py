from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_OUTPUT_DIR = "src/embed/output/dense"
DOCUMENT_PREFIX = "passage: "
QUERY_PREFIX = "query: "


class Encoder(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int | None = None,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class PreparedChunk:
    chunk_id: str
    article_id: str
    embedding_input: str
    metadata: dict[str, object]
    text_chars: int
    estimated_tokens: int


@dataclass
class LengthStats:
    min: int = 0
    max: int = 0
    avg: float = 0.0


@dataclass
class EmbeddingStats:
    chunks_read: int
    chunks_embedded: int
    skipped_empty_text: int
    embedding_dimension: int
    batch_size: int
    elapsed_seconds: float
    chunks_per_second: float | None
    token_stats: LengthStats
    char_stats: LengthStats
    strategy_counts: dict[str, int]
    implementation_counts: dict[str, int]
    category_counts: dict[str, int]
    longest_chunks: list[dict[str, object]]


def ensure_prefix(text: str, prefix: str) -> str:
    value = text.strip()
    return value if value.startswith(prefix) else f"{prefix}{value}"


def prepare_document_text(text: str) -> str:
    return ensure_prefix(text, DOCUMENT_PREFIX)


def prepare_query_text(text: str, prefix: str | None = None) -> str:
    pref = QUERY_PREFIX if prefix is None else prefix
    return ensure_prefix(text, pref)


def detect_prefixes(model_name: str, doc_pref: str | None = None, query_pref: str | None = None) -> tuple[str, str]:
    if doc_pref is not None and query_pref is not None:
        return doc_pref, query_pref

    model_lower = model_name.lower()
    
    # E5 auto-detect
    if "e5" in model_lower:
        d = "passage: " if doc_pref is None else doc_pref
        q = "query: " if query_pref is None else query_pref
        return d, q
        
    # Qwen auto-detect
    if "qwen" in model_lower:
        d = "" if doc_pref is None else doc_pref
        q = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: " if query_pref is None else query_pref
        return d, q

    # Default fallback: no prefix (for BGE, GTE or others)
    d = "" if doc_pref is None else doc_pref
    q = "" if query_pref is None else query_pref
    return d, q



def estimate_tokens(text: str) -> int:
    return len(text.split())


def read_chunk_jsonl(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prepare_chunks(rows: Iterable[dict[str, object]], document_prefix: str = DOCUMENT_PREFIX) -> tuple[list[PreparedChunk], int]:
    prepared: list[PreparedChunk] = []
    skipped_empty_text = 0
    for row in rows:
        raw_text = str(row.get("text") or "").strip()
        if not raw_text:
            skipped_empty_text += 1
            continue

        metadata = dict(row.get("metadata") or {})
        strategy = str(row.get("strategy") or metadata.get("strategy") or "")
        if strategy:
            metadata["strategy"] = strategy
        metadata["chunk_id"] = str(row["chunk_id"])
        metadata["article_id"] = str(row["article_id"])
        metadata["chunk_text"] = str(row.get("chunk_text") or "")
        metadata["text"] = raw_text

        embedding_input = ensure_prefix(raw_text, document_prefix)
        prepared.append(
            PreparedChunk(
                chunk_id=str(row["chunk_id"]),
                article_id=str(row["article_id"]),
                embedding_input=embedding_input,
                metadata=metadata,
                text_chars=len(embedding_input),
                estimated_tokens=estimate_tokens(embedding_input),
            )
        )
    return prepared, skipped_empty_text


def encode_prepared_chunks(
    prepared_chunks: list[PreparedChunk],
    encoder: Encoder,
    *,
    batch_size: int,
    normalize_embeddings: bool,
    show_progress: bool,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    iterator = range(0, len(prepared_chunks), batch_size)
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, total=(len(prepared_chunks) + batch_size - 1) // batch_size, desc="Embedding", unit="batch")

    for start in iterator:
        batch = prepared_chunks[start : start + batch_size]
        vectors = encoder.encode(
            [item.embedding_input for item in batch],
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
        )
        embeddings.append(np.asarray(vectors, dtype=np.float32))

    if not embeddings:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(embeddings)


def build_stats(
    *,
    prepared_chunks: list[PreparedChunk],
    skipped_empty_text: int,
    embeddings: np.ndarray,
    batch_size: int,
    elapsed_seconds: float,
) -> EmbeddingStats:
    token_lengths = [item.estimated_tokens for item in prepared_chunks]
    char_lengths = [item.text_chars for item in prepared_chunks]
    dimension = int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[0] else 0
    chunks_per_second = round(len(prepared_chunks) / elapsed_seconds, 4) if elapsed_seconds > 0 else None

    return EmbeddingStats(
        chunks_read=len(prepared_chunks) + skipped_empty_text,
        chunks_embedded=len(prepared_chunks),
        skipped_empty_text=skipped_empty_text,
        embedding_dimension=dimension,
        batch_size=batch_size,
        elapsed_seconds=round(elapsed_seconds, 6),
        chunks_per_second=chunks_per_second,
        token_stats=_length_stats(token_lengths),
        char_stats=_length_stats(char_lengths),
        strategy_counts=_count_metadata(prepared_chunks, "strategy"),
        implementation_counts=_count_metadata(prepared_chunks, "implementation"),
        category_counts=_count_metadata(prepared_chunks, "category"),
        longest_chunks=_longest_chunks(prepared_chunks, limit=10),
    )


def _length_stats(values: list[int]) -> LengthStats:
    if not values:
        return LengthStats()
    return LengthStats(min=min(values), max=max(values), avg=round(sum(values) / len(values), 4))


def _count_metadata(prepared_chunks: list[PreparedChunk], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in prepared_chunks:
        value = str(item.metadata.get(key) or "(missing)")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))


def _longest_chunks(prepared_chunks: list[PreparedChunk], *, limit: int) -> list[dict[str, object]]:
    longest = sorted(prepared_chunks, key=lambda item: item.estimated_tokens, reverse=True)[:limit]
    return [
        {
            "chunk_id": item.chunk_id,
            "article_id": item.article_id,
            "strategy": item.metadata.get("strategy"),
            "estimated_tokens": item.estimated_tokens,
            "text_chars": item.text_chars,
            "title": item.metadata.get("title"),
            "category": item.metadata.get("category"),
        }
        for item in longest
    ]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def write_outputs(
    *,
    output_dir: str | Path,
    input_path: str | Path,
    model_name: str,
    batch_size: int,
    normalize_embeddings: bool,
    prepared_chunks: list[PreparedChunk],
    embeddings: np.ndarray,
    stats: EmbeddingStats,
    sample_size: int,
    document_prefix: str = DOCUMENT_PREFIX,
    query_prefix: str = QUERY_PREFIX,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    embeddings_path = output / "embeddings.npy"
    metadata_path = output / "metadata.jsonl"
    manifest_path = output / "manifest.json"
    stats_path = output / "embedding_stats.json"
    samples_path = output / "debug_samples.jsonl"

    np.save(embeddings_path, embeddings)
    write_jsonl(metadata_path, (item.metadata for item in prepared_chunks))
    stats_payload = _to_plain_dict(stats)
    stats_path.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(samples_path, build_debug_samples(prepared_chunks, sample_size=sample_size))

    manifest = {
        "model_name": model_name,
        "embedding_dimension": stats.embedding_dimension,
        "normalize_embeddings": normalize_embeddings,
        "input_path": str(input_path),
        "output_dir": str(output),
        "embeddings_path": str(embeddings_path),
        "metadata_path": str(metadata_path),
        "stats_path": str(stats_path),
        "debug_samples_path": str(samples_path),
        "batch_size": batch_size,
        "chunk_count": stats.chunks_embedded,
        "elapsed_seconds": stats.elapsed_seconds,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "document_prefix": document_prefix,
        "query_prefix": query_prefix,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "embeddings": str(embeddings_path),
        "metadata": str(metadata_path),
        "manifest": str(manifest_path),
        "stats": str(stats_path),
        "debug_samples": str(samples_path),
    }


def build_debug_samples(prepared_chunks: list[PreparedChunk], *, sample_size: int) -> list[dict[str, object]]:
    samples = prepared_chunks[: max(0, sample_size)]
    return [
        {
            "chunk_id": item.chunk_id,
            "article_id": item.article_id,
            "strategy": item.metadata.get("strategy"),
            "implementation": item.metadata.get("implementation"),
            "chunk_index": item.metadata.get("chunk_index"),
            "title": item.metadata.get("title"),
            "category": item.metadata.get("category"),
            "estimated_tokens": item.estimated_tokens,
            "text_chars": item.text_chars,
            "embedding_input_preview": _preview(item.embedding_input, 500),
            "chunk_text_preview": _preview(str(item.metadata.get("chunk_text") or ""), 500),
        }
        for item in samples
    ]


def _preview(text: str, limit: int) -> str:
    value = " ".join(text.split())
    return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."


def _to_plain_dict(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _to_plain_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_dict(item) for item in value]
    return value


def load_sentence_transformer(model_name: str) -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    # Cap max sequence length to 512 to avoid CUDA OOM on long inputs
    if hasattr(model, "max_seq_length") and model.max_seq_length > 512:
        model.max_seq_length = 512
    return model


def embed_chunks_file(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    model_name: str,
    batch_size: int,
    normalize_embeddings: bool = True,
    sample_size: int,
    show_progress: bool,
    encoder: Encoder | None = None,
    document_prefix: str | None = None,
    query_prefix: str | None = None,
) -> tuple[EmbeddingStats, dict[str, str], list[dict[str, object]]]:
    doc_pref, query_pref = detect_prefixes(model_name, document_prefix, query_prefix)
    rows = read_chunk_jsonl(input_path)
    prepared_chunks, skipped_empty_text = prepare_chunks(rows, document_prefix=doc_pref)
    active_encoder = encoder or load_sentence_transformer(model_name)

    started = time.perf_counter()
    embeddings = encode_prepared_chunks(
        prepared_chunks,
        active_encoder,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress=show_progress,
    )
    elapsed = time.perf_counter() - started
    stats = build_stats(
        prepared_chunks=prepared_chunks,
        skipped_empty_text=skipped_empty_text,
        embeddings=embeddings,
        batch_size=batch_size,
        elapsed_seconds=elapsed,
    )
    output_paths = write_outputs(
        output_dir=output_dir,
        input_path=input_path,
        model_name=model_name,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        prepared_chunks=prepared_chunks,
        embeddings=embeddings,
        stats=stats,
        sample_size=sample_size,
        document_prefix=doc_pref,
        query_prefix=query_pref,
    )
    return stats, output_paths, build_debug_samples(prepared_chunks, sample_size=sample_size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed chunk JSONL files for dense retrieval experiments.")
    parser.add_argument("--input", required=True, help="Input chunk JSONL from src.chunking.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for embeddings and metadata.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--show-samples", action="store_true", help="Print debug samples after embedding.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable embedding normalization.")
    parser.add_argument("--document-prefix", default=None, help="Prefix prepended to chunks. Auto-detects if omitted.")
    parser.add_argument("--query-prefix", default=None, help="Prefix prepended to queries. Auto-detects if omitted.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats, output_paths, samples = embed_chunks_file(
        input_path=args.input,
        output_dir=args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        normalize_embeddings=not args.no_normalize,
        sample_size=args.sample_size,
        show_progress=not args.no_progress,
        document_prefix=args.document_prefix,
        query_prefix=args.query_prefix,
    )
    print(json.dumps({"stats": _to_plain_dict(stats), "outputs": output_paths}, ensure_ascii=False, indent=2))
    if args.show_samples:
        print(json.dumps({"debug_samples": samples}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



