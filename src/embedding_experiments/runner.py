from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

try:
    from src.chunking.strategies import DEFAULT_STRATEGIES, read_jsonl
    from src.embedding_experiments.data import load_chunks, load_queries, preprocess_corpus_csvs, preprocess_corpus_jsonl
    from src.embedding_experiments.metrics import average_metrics, retrieval_metrics
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.chunking.strategies import DEFAULT_STRATEGIES, read_jsonl
    from src.embedding_experiments.data import load_chunks, load_queries, preprocess_corpus_csvs, preprocess_corpus_jsonl
    from src.embedding_experiments.metrics import average_metrics, retrieval_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BAAI/bge-m3 dense retrieval experiments.")
    parser.add_argument(
        "--corpus-csv",
        nargs="+",
        default=[
            "Dataset/Create_QA_Vietonline/VietOnlineNews/train_new.csv",
            "Dataset/Create_QA_Vietonline/VietOnlineNews/validation_new.csv",
            "Dataset/Create_QA_Vietonline/VietOnlineNews/test_new.csv",
        ],
        help="Corpus CSV files to index. Defaults to all VietOnlineNews *_new splits.",
    )
    parser.add_argument(
        "--corpus-jsonl",
        default=None,
        help="Raw corpus JSONL file to index. If set, this overrides --corpus-csv.",
    )
    parser.add_argument("--qa-csv", default="Dataset/QA_Claude/QA_output.csv")
    parser.add_argument("--work-dir", default="reports/embedding_bge_m3")
    parser.add_argument(
        "--chunks-dir",
        default=None,
        help="Directory containing precomputed chunk files named chunks_<strategy>.jsonl. Defaults to --work-dir.",
    )
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES), choices=list(DEFAULT_STRATEGIES))
    parser.add_argument("--chunk-size", type=int, default=450)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--min-chunk-tokens", type=int, default=80)
    parser.add_argument("--small-article-chars", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--article-limit", type=int, default=None)
    parser.add_argument("--device", default=None, help="Example: cuda, cpu. Default lets sentence-transformers choose.")
    parser.add_argument("--query-prefix", default="", help="Optional text prepended to every query before encoding.")
    parser.add_argument("--passage-prefix", default="", help="Optional text prepended to every chunk before encoding.")
    parser.add_argument("--skip-eval", action="store_true", help="Only chunk and embed; do not load QA or compute metrics.")
    parser.add_argument("--save-embeddings", action="store_true", default=True, help="Save .npy embeddings and metadata JSONL.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild preprocessed corpus and chunk files.")
    parser.add_argument("--normalize-embeddings", action="store_true", default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    total_started = time.perf_counter()
    root = Path.cwd()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = Path(args.chunks_dir) if args.chunks_dir else work_dir
    corpus_jsonl = work_dir / "vieonline_news_clean_all.jsonl"

    stage_times: dict[str, float] = {}
    started = time.perf_counter()
    if args.force_rebuild or not corpus_jsonl.exists():
        if args.corpus_jsonl:
            preprocess_stats = preprocess_corpus_jsonl(args.corpus_jsonl, corpus_jsonl)
        else:
            preprocess_stats = preprocess_corpus_csvs(args.corpus_csv, corpus_jsonl)
    else:
        preprocess_stats = {"reused": 1, "path": str(corpus_jsonl)}
    stage_times["preprocess_seconds"] = round(time.perf_counter() - started, 6)

    queries = [] if args.skip_eval else load_queries(args.qa_csv, limit=args.query_limit)
    if not args.skip_eval and not queries:
        raise ValueError(f"No answerable queries found in {args.qa_csv}")

    model, load_model_seconds = load_embedding_model(args.model, device=args.device)
    stage_times["load_model_seconds"] = load_model_seconds

    all_results = []
    per_strategy_payload = {}
    for strategy in args.strategies:
        strategy_started = time.perf_counter()
        chunk_path = chunks_dir / f"chunks_{strategy}.jsonl"
        if not chunk_path.exists():
            raise FileNotFoundError(
                f"Missing precomputed chunk file for strategy={strategy}: {chunk_path}. "
                "Run chunking first or pass --chunks-dir to the folder containing prepared chunks."
            )
        chunk_stats = {"elapsed_seconds": 0.0, "reused": 1}
        chunks = load_chunks(chunk_path)
        if not chunks:
            raise ValueError(f"No chunks produced for strategy={strategy}")

        texts = [args.passage_prefix + str(chunk["text"]) for chunk in chunks]
        article_ids = [str(chunk["article_id"]) for chunk in chunks]
        chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
        token_counts = [int(chunk.get("metadata", {}).get("token_count", 0)) for chunk in chunks]

        encode_started = time.perf_counter()
        chunk_embeddings = encode_texts(
            model,
            texts,
            batch_size=args.batch_size,
            normalize_embeddings=args.normalize_embeddings,
        )
        embedding_time_seconds = round(time.perf_counter() - encode_started, 6)
        if args.save_embeddings:
            save_embeddings(work_dir, strategy, chunk_embeddings, chunks)

        if args.skip_eval:
            evaluation_seconds = 0.0
            metrics = empty_retrieval_metrics()
            latency_avg = 0.0
            latency_p95 = 0.0
            sample_results = []
        else:
            eval_started = time.perf_counter()
            query_metrics, latencies, sample_results = evaluate_queries(
                model=model,
                queries=queries,
                chunk_embeddings=chunk_embeddings,
                chunk_ids=chunk_ids,
                article_ids=article_ids,
                top_k=args.top_k,
                batch_size=args.batch_size,
                query_prefix=args.query_prefix,
                normalize_embeddings=args.normalize_embeddings,
            )
            evaluation_seconds = round(time.perf_counter() - eval_started, 6)
            metrics = average_metrics(query_metrics)
            latency_avg = statistics.fmean(latencies) if latencies else 0.0
            latency_p95 = percentile(latencies, 95)

        result = {
            "model": args.model,
            "chunking": strategy,
            "retrieval_type": "dense",
            **metrics,
            "embedding_time_seconds": embedding_time_seconds,
            "chunks_per_second": round(len(chunks) / embedding_time_seconds, 6) if embedding_time_seconds else None,
            "query_latency_ms_avg": round(latency_avg, 6),
            "query_latency_ms_p95": round(latency_p95, 6),
            "index_size_mb": round(chunk_embeddings.nbytes / (1024 * 1024), 6),
            "embedding_dimension": int(chunk_embeddings.shape[1]),
            "num_chunks": len(chunks),
            "avg_chunk_tokens": round(statistics.fmean(token_counts), 6) if token_counts else 0.0,
            "num_queries": len(queries),
            "chunking_time_seconds": chunk_stats["elapsed_seconds"],
            "evaluation_seconds": evaluation_seconds,
            "strategy_total_seconds": round(time.perf_counter() - strategy_started, 6),
        }
        all_results.append(result)
        per_strategy_payload[strategy] = {
            "result": result,
            "chunk_stats": chunk_stats,
            "sample_results": sample_results[:20],
        }
        write_json(work_dir / f"results_{strategy}.json", per_strategy_payload[strategy])
        print(json.dumps(result, ensure_ascii=False, indent=2))

    total_runtime_seconds = round(time.perf_counter() - total_started, 6)
    stage_times["total_runtime_seconds"] = total_runtime_seconds
    write_csv(work_dir / "leaderboard.csv", all_results)
    write_json(
        work_dir / "experiment_summary.json",
        {
            "model": args.model,
            "corpus_csv": args.corpus_csv if not args.corpus_jsonl else None,
            "corpus_jsonl": args.corpus_jsonl,
            "qa_csv": args.qa_csv,
            "chunks_dir": str(chunks_dir),
            "preprocess_stats": preprocess_stats,
            "stage_times": stage_times,
            "leaderboard": sorted(all_results, key=lambda item: item["ndcg@10"], reverse=True),
        },
    )
    (work_dir / "leaderboard.md").write_text(build_markdown(all_results, stage_times), encoding="utf-8")
    (work_dir / "notes.md").write_text(build_notes(args, stage_times), encoding="utf-8")
    print(f"Total runtime: {total_runtime_seconds}s")
    print(f"Reports written to: {work_dir.resolve()}")


def load_embedding_model(model_name: str, *, device: str | None):
    started = time.perf_counter()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Install sentence-transformers before running dense embedding experiments.") from exc
    model = SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)
    return model, round(time.perf_counter() - started, 6)


def encode_texts(model, texts: list[str], *, batch_size: int, normalize_embeddings: bool) -> np.ndarray:
    import numpy as np

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    return np.asarray(embeddings, dtype=np.float32)


def save_embeddings(work_dir: Path, strategy: str, embeddings, chunks: list[dict[str, object]]) -> None:
    import numpy as np

    np.save(work_dir / f"embeddings_{strategy}.npy", embeddings)
    metadata_path = work_dir / f"embedding_metadata_{strategy}.jsonl"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        for chunk in chunks:
            payload = {
                "chunk_id": chunk.get("chunk_id"),
                "article_id": chunk.get("article_id"),
                "strategy": chunk.get("strategy"),
                "metadata": chunk.get("metadata"),
            }
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def empty_retrieval_metrics() -> dict[str, float]:
    return {
        "ndcg@10": 0.0,
        "recall@5": 0.0,
        "recall@10": 0.0,
        "mrr@10": 0.0,
        "hit@1": 0.0,
        "hit@5": 0.0,
    }


def evaluate_queries(
    *,
    model,
    queries,
    chunk_embeddings: np.ndarray,
    chunk_ids: list[str],
    article_ids: list[str],
    top_k: int,
    batch_size: int,
    query_prefix: str,
    normalize_embeddings: bool,
) -> tuple[list[dict[str, float]], list[float], list[dict[str, object]]]:
    import numpy as np

    metrics_rows: list[dict[str, float]] = []
    latencies: list[float] = []
    samples: list[dict[str, object]] = []
    for query in queries:
        started = time.perf_counter()
        query_embedding = model.encode(
            [query_prefix + query.question],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )
        scores = chunk_embeddings @ np.asarray(query_embedding[0], dtype=np.float32)
        top_indices = top_indices_desc(scores, top_k)
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        ranked_article_ids = [article_ids[index] for index in top_indices]
        metrics_rows.append(retrieval_metrics(unique_preserve_order(ranked_article_ids), query.relevant_article_ids, max_k=top_k))
        samples.append(
            {
                "query_id": query.query_id,
                "question": query.question,
                "relevant_article_ids": sorted(query.relevant_article_ids),
                "results": [
                    {
                        "rank": rank,
                        "chunk_id": chunk_ids[index],
                        "article_id": article_ids[index],
                        "score": round(float(scores[index]), 6),
                    }
                    for rank, index in enumerate(top_indices, start=1)
                ],
            }
        )
    return metrics_rows, latencies, samples


def top_indices_desc(scores: np.ndarray, top_k: int) -> list[int]:
    import numpy as np

    k = min(top_k, len(scores))
    if k <= 0:
        return []
    candidate_indices = np.argpartition(scores, -k)[-k:]
    return candidate_indices[np.argsort(scores[candidate_indices])[::-1]].tolist()


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def percentile(values: list[float], percent: int) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percent / 100
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(rows: list[dict[str, object]], stage_times: dict[str, float]) -> str:
    ordered = sorted(rows, key=lambda item: item["ndcg@10"], reverse=True)
    lines = [
        "# BAAI/bge-m3 Embedding Leaderboard",
        "",
        f"Total runtime: `{stage_times['total_runtime_seconds']}s`",
        "",
        "| Rank | Model | Chunking | nDCG@10 | Recall@10 | MRR@10 | Latency avg ms | Index MB |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(ordered, start=1):
        lines.append(
            f"| {rank} | {row['model']} | {row['chunking']} | {row['ndcg@10']} | "
            f"{row['recall@10']} | {row['mrr@10']} | {row['query_latency_ms_avg']} | {row['index_size_mb']} |"
        )
    return "\n".join(lines) + "\n"


def build_notes(args, stage_times: dict[str, float]) -> str:
    return "\n".join(
        [
            "# Notes - Tuấn Anh / BAAI/bge-m3",
            "",
            f"- Model: `{args.model}`",
            f"- Retrieval: dense cosine/dot product over normalized embeddings.",
            f"- Query prefix/instruction: `{args.query_prefix or '(none)'}`",
            f"- Passage prefix/instruction: `{args.passage_prefix or '(none)'}`",
            f"- Batch size: `{args.batch_size}`",
            f"- Device: `{args.device or 'sentence-transformers default'}`",
            f"- Chunk size / overlap: `{args.chunk_size}` / `{args.overlap}`",
            f"- Total runtime seconds: `{stage_times['total_runtime_seconds']}`",
            "",
            "Ranking chính dùng `nDCG@10`; khi gần nhau ưu tiên `Recall@10` cao hơn và latency thấp hơn.",
        ]
    ) + "\n"


if __name__ == "__main__":
    main()
