"""Embedding retrieval experiment harness.

Encodes each chunking strategy with a sentence-transformers model, runs the QA
queries against a cosine-similarity index, and reports retrieval + efficiency
metrics so configurations can be ranked by nDCG@10.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Set

import numpy as np

try:
    from .metrics import evaluate_query
except ImportError:  # Allow `python src\embedding\experiment.py`
    from metrics import evaluate_query


DEFAULT_STRATEGIES = ("token", "langchain_recursive", "llamaindex", "structured")


@dataclass
class Chunk:
    chunk_id: str
    article_id: str
    text: str


@dataclass
class QAItem:
    qa_id: str
    question: str
    gold_articles: Set[str]
    is_possible: bool
    qa_type: str


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_chunks(path: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    for row in read_jsonl(path):
        text = row.get("text") or row.get("chunk_text") or ""
        if not text.strip():
            continue
        chunks.append(
            Chunk(
                chunk_id=str(row["chunk_id"]),
                article_id=str(row["article_id"]),
                text=text,
            )
        )
    return chunks


def _gold_articles(row: dict) -> Set[str]:
    out: Set[str] = set()
    art = row.get("article_id")
    if isinstance(art, list):
        out.update(str(x) for x in art)
    elif art is not None:
        out.add(str(art))
    src = row.get("source_article_ids")
    if isinstance(src, list):
        out.update(str(x) for x in src)
    return out


def load_qa(path: Path, include_unanswerable: bool = False) -> List[QAItem]:
    items: List[QAItem] = []
    for row in read_jsonl(path):
        is_possible = bool(row.get("is_possible", True))
        gold = _gold_articles(row)
        if not is_possible and not include_unanswerable:
            continue
        if not gold:
            continue
        items.append(
            QAItem(
                qa_id=str(row.get("id")),
                question=str(row.get("question", "")),
                gold_articles=gold,
                is_possible=is_possible,
                qa_type=str(row.get("qa_type", "")),
            )
        )
    return items


@dataclass
class StrategyResult:
    strategy: str
    model: str
    embedding_dimension: int
    num_chunks: int
    avg_chunk_tokens: float
    embedding_time_seconds: float
    chunks_per_second: float
    index_size_mb: float
    query_latency_ms_avg: float
    query_latency_ms_p95: float
    metrics: Dict[str, float]
    per_query: List[dict] = field(default_factory=list)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def run_strategy(
    strategy: str,
    chunk_path: Path,
    qa_items: Sequence[QAItem],
    model,
    model_name: str,
    batch_size: int,
    top_k: int,
    query_prefix: str = "",
    doc_prefix: str = "",
) -> StrategyResult:
    chunks = load_chunks(chunk_path)
    chunk_articles = np.array([c.article_id for c in chunks])
    doc_texts = [doc_prefix + c.text for c in chunks]

    start = time.perf_counter()
    doc_embeddings = model.encode(
        doc_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    embedding_time = time.perf_counter() - start
    doc_embeddings = _normalize(doc_embeddings)

    dim = int(doc_embeddings.shape[1])
    index_size_mb = doc_embeddings.nbytes / (1024 * 1024)
    avg_tokens = float(np.mean([len(c.text.split()) for c in chunks])) if chunks else 0.0

    query_texts = [query_prefix + item.question for item in qa_items]
    query_embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    ).astype(np.float32)
    query_embeddings = _normalize(query_embeddings)

    agg: Dict[str, float] = {}
    latencies: List[float] = []
    per_query: List[dict] = []

    for item, q_vec in zip(qa_items, query_embeddings):
        t0 = time.perf_counter()
        sims = doc_embeddings @ q_vec
        k = min(top_k, len(sims))
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        latencies.append((time.perf_counter() - t0) * 1000.0)

        ranked_articles = [chunk_articles[i] for i in top_idx]
        scores = evaluate_query(ranked_articles, item.gold_articles)
        for key, val in scores.items():
            agg[key] = agg.get(key, 0.0) + val
        per_query.append(
            {
                "qa_id": item.qa_id,
                "qa_type": item.qa_type,
                "gold_articles": sorted(item.gold_articles),
                "top_articles": ranked_articles[:10],
                **scores,
            }
        )

    n = max(len(qa_items), 1)
    metrics = {key: val / n for key, val in agg.items()}
    latencies_sorted = sorted(latencies)
    if latencies_sorted:
        p95 = latencies_sorted[min(len(latencies_sorted) - 1, int(0.95 * len(latencies_sorted)))]
    else:
        p95 = 0.0

    return StrategyResult(
        strategy=strategy,
        model=model_name,
        embedding_dimension=dim,
        num_chunks=len(chunks),
        avg_chunk_tokens=round(avg_tokens, 2),
        embedding_time_seconds=round(embedding_time, 3),
        chunks_per_second=round(len(chunks) / embedding_time, 2) if embedding_time > 0 else 0.0,
        index_size_mb=round(index_size_mb, 3),
        query_latency_ms_avg=round(float(np.mean(latencies)), 3) if latencies else 0.0,
        query_latency_ms_p95=round(float(p95), 3),
        metrics={k: round(v, 4) for k, v in metrics.items()},
        per_query=per_query,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run embedding retrieval experiments per chunking strategy.")
    parser.add_argument("--model", default="Alibaba-NLP/gte-multilingual-base")
    parser.add_argument("--chunk-dir", default="src/chunking/output")
    parser.add_argument("--qa", default="Dataset/QA_Claude/QA_output.jsonl")
    parser.add_argument("--output-dir", default="src/embedding/output")
    parser.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--doc-prefix", default="")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--limit-qa", type=int, default=None)
    parser.add_argument("--no-trust-remote-code", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from sentence_transformers import SentenceTransformer

    chunk_dir = Path(args.chunk_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qa_items = load_qa(Path(args.qa))
    if args.limit_qa:
        qa_items = qa_items[: args.limit_qa]
    print(f"Loaded {len(qa_items)} answerable QA queries with gold articles.")

    print(f"Loading model {args.model} ...")
    model = SentenceTransformer(args.model, trust_remote_code=not args.no_trust_remote_code)
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length

    results: List[StrategyResult] = []
    for strategy in args.strategies:
        chunk_path = chunk_dir / f"vieonline_news_chunks_{strategy}.jsonl"
        if not chunk_path.exists():
            print(f"[skip] {chunk_path} not found")
            continue
        print(f"\n=== Strategy: {strategy} ===")
        result = run_strategy(
            strategy=strategy,
            chunk_path=chunk_path,
            qa_items=qa_items,
            model=model,
            model_name=args.model,
            batch_size=args.batch_size,
            top_k=args.top_k,
            query_prefix=args.query_prefix,
            doc_prefix=args.doc_prefix,
        )
        results.append(result)
        print(json.dumps({"strategy": strategy, **result.metrics}, ensure_ascii=False))

    summary = []
    for r in results:
        summary.append(
            {
                "strategy": r.strategy,
                "model": r.model,
                "num_chunks": r.num_chunks,
                "avg_chunk_tokens": r.avg_chunk_tokens,
                "embedding_dimension": r.embedding_dimension,
                "embedding_time_seconds": r.embedding_time_seconds,
                "chunks_per_second": r.chunks_per_second,
                "index_size_mb": r.index_size_mb,
                "query_latency_ms_avg": r.query_latency_ms_avg,
                "query_latency_ms_p95": r.query_latency_ms_p95,
                **r.metrics,
            }
        )
        per_q_path = out_dir / f"per_query_{r.strategy}.jsonl"
        with per_q_path.open("w", encoding="utf-8") as handle:
            for row in r.per_query:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    (out_dir / "embedding_summary.json").write_text(
        json.dumps({"model": args.model, "num_queries": len(qa_items), "results": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_report(out_dir / "embedding_report.md", args.model, len(qa_items), summary)
    print(f"\nWrote summary and report to {out_dir}")


def write_report(path: Path, model: str, num_queries: int, summary: List[dict]) -> None:
    ranked = sorted(
        summary,
        key=lambda r: (r.get("ndcg@10", 0.0), r.get("recall@10", 0.0), -r.get("query_latency_ms_avg", 0.0)),
        reverse=True,
    )
    lines: List[str] = []
    lines.append("# Embedding Experiment Report")
    lines.append("")
    lines.append(f"- Model: `{model}`")
    lines.append(f"- Answerable queries with gold articles: {num_queries}")
    lines.append("- Relevance: article-level (top-k chunks mapped to article_id)")
    lines.append("- Ranking metric: nDCG@10 (tie-break: Recall@10, then lower latency)")
    lines.append("")
    lines.append("## Leaderboard")
    lines.append("")
    lines.append("| Rank | Strategy | nDCG@10 | Recall@10 | Recall@5 | MRR@10 | Hit@1 | Hit@5 | Latency avg ms | Index MB |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(ranked, start=1):
        lines.append(
            f"| {i} | {r['strategy']} | {r.get('ndcg@10',0):.4f} | {r.get('recall@10',0):.4f} | "
            f"{r.get('recall@5',0):.4f} | {r.get('mrr@10',0):.4f} | {r.get('hit@1',0):.4f} | "
            f"{r.get('hit@5',0):.4f} | {r['query_latency_ms_avg']:.2f} | {r['index_size_mb']:.2f} |"
        )
    lines.append("")
    lines.append("## Efficiency & Index")
    lines.append("")
    lines.append("| Strategy | num_chunks | avg_chunk_tokens | embed_dim | embed_time_s | chunks/s | index_MB |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in summary:
        lines.append(
            f"| {r['strategy']} | {r['num_chunks']} | {r['avg_chunk_tokens']:.1f} | {r['embedding_dimension']} | "
            f"{r['embedding_time_seconds']:.2f} | {r['chunks_per_second']:.1f} | {r['index_size_mb']:.2f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()