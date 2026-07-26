"""Fuse the existing dense retrieval output with an in-process BM25 retriever."""
from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embedding.metrics import evaluate_query


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_chunks(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for index, row in enumerate(read_jsonl(path)):
        if limit is not None and index >= limit:
            break
        text = str(row.get("text") or row.get("chunk_text") or "").strip()
        if not text:
            continue
        chunks.append(
            {
                "chunk_index": index,
                "chunk_id": str(row.get("chunk_id", index)),
                "article_id": str(row.get("article_id", "")),
                "text": text,
            }
        )
    return chunks


def query_vocabulary(rows: Sequence[Dict[str, Any]]) -> Set[str]:
    vocabulary: Set[str] = set()
    for row in rows:
        vocabulary.update(tokenize(str(row.get("question", ""))))
    return vocabulary


class BM25Index:
    """Memory-conscious BM25 index containing postings only for evaluation query terms."""

    def __init__(
        self,
        chunks: Sequence[Dict[str, Any]],
        vocabulary: Set[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_lengths: List[int] = []
        self.postings: DefaultDict[str, List[Tuple[int, int]]] = defaultdict(list)
        document_frequency: Counter[str] = Counter()

        for doc_index, chunk in enumerate(chunks):
            document_tokens = tokenize(str(chunk["text"]))
            self.doc_lengths.append(len(document_tokens))
            matching_counts = Counter(
                token for token in document_tokens if token in vocabulary
            )
            for term, frequency in matching_counts.items():
                self.postings[term].append((doc_index, frequency))
                document_frequency[term] += 1

        self.num_docs = len(chunks)
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.num_docs if self.num_docs else 0.0
        )
        self.idf = {
            term: math.log(
                1.0 + (self.num_docs - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if self.num_docs == 0 or top_k <= 0:
            return []
        scores: DefaultDict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_index, term_frequency in self.postings[term]:
                length_ratio = self.doc_lengths[doc_index] / max(self.avg_doc_length, 1.0)
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * length_ratio
                )
                scores[doc_index] += (
                    idf * term_frequency * (self.k1 + 1.0) / denominator
                )
        return heapq.nlargest(
            min(top_k, len(scores)),
            scores.items(),
            key=lambda item: item[1],
        )


def minmax_scores(candidates: Sequence[Dict[str, Any]], score_key: str) -> Dict[str, float]:
    if not candidates:
        return {}
    scores = [float(candidate.get(score_key, 0.0)) for candidate in candidates]
    low, high = min(scores), max(scores)
    if math.isclose(low, high):
        return {str(candidate["chunk_id"]): 1.0 for candidate in candidates}
    return {
        str(candidate["chunk_id"]): (float(candidate.get(score_key, 0.0)) - low) / (high - low)
        for candidate in candidates
    }


def fuse_candidates(
    dense_candidates: Sequence[Dict[str, Any]],
    bm25_candidates: Sequence[Dict[str, Any]],
    method: str,
    top_k: int,
    dense_weight: float,
    bm25_weight: float,
    rrf_k: int,
) -> List[Dict[str, Any]]:
    pool: Dict[str, Dict[str, Any]] = {}
    for candidate in list(dense_candidates) + list(bm25_candidates):
        chunk_id = str(candidate.get("chunk_id", ""))
        if chunk_id and chunk_id not in pool:
            pool[chunk_id] = {
                "chunk_index": candidate.get("chunk_index"),
                "chunk_id": chunk_id,
                "article_id": str(candidate.get("article_id", "")),
                "text": candidate.get("text", ""),
            }

    dense_normalized = minmax_scores(dense_candidates, "dense_score")
    bm25_normalized = minmax_scores(bm25_candidates, "bm25_score")

    for dense_rank, candidate in enumerate(dense_candidates, start=1):
        chunk_id = str(candidate["chunk_id"])
        pool[chunk_id]["dense_rank"] = dense_rank
        pool[chunk_id]["dense_score"] = float(candidate.get("dense_score", 0.0))
    for bm25_rank, candidate in enumerate(bm25_candidates, start=1):
        chunk_id = str(candidate["chunk_id"])
        pool[chunk_id]["bm25_rank"] = bm25_rank
        pool[chunk_id]["bm25_score"] = float(candidate.get("bm25_score", 0.0))

    for chunk_id, candidate in pool.items():
        if method == "rrf":
            dense_part = (
                dense_weight / (rrf_k + int(candidate["dense_rank"]))
                if candidate.get("dense_rank")
                else 0.0
            )
            bm25_part = (
                bm25_weight / (rrf_k + int(candidate["bm25_rank"]))
                if candidate.get("bm25_rank")
                else 0.0
            )
            candidate["fusion_score"] = dense_part + bm25_part
        else:
            candidate["fusion_score"] = (
                dense_weight * dense_normalized.get(chunk_id, 0.0)
                + bm25_weight * bm25_normalized.get(chunk_id, 0.0)
            )

    ranked = sorted(pool.values(), key=lambda item: item["fusion_score"], reverse=True)[:top_k]
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
        candidate["score"] = round(float(candidate["fusion_score"]), 8)
        candidate["fusion_score"] = round(float(candidate["fusion_score"]), 8)
    return ranked


def dense_candidates_from_row(row: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    source = row.get("reranked_candidates") or row.get("candidates") or []
    candidates: List[Dict[str, Any]] = []
    for candidate in source[:top_k]:
        dense_score = candidate.get(
            "score",
            candidate.get("retrieval_score", candidate.get("rerank_score", 0.0)),
        )
        candidates.append(
            {
                "chunk_index": candidate.get("chunk_index"),
                "chunk_id": str(candidate.get("chunk_id", "")),
                "article_id": str(candidate.get("article_id", "")),
                "text": candidate.get("text", ""),
                "dense_score": float(dense_score or 0.0),
            }
        )
    return candidates


def build_output_row(
    row: Dict[str, Any],
    index: BM25Index,
    method: str,
    dense_top_k: int,
    bm25_top_k: int,
    output_top_k: int,
    dense_weight: float,
    bm25_weight: float,
    rrf_k: int,
) -> Dict[str, Any]:
    dense_candidates = dense_candidates_from_row(row, dense_top_k)
    bm25_hits = index.search(str(row.get("question", "")), bm25_top_k)
    bm25_candidates = [
        {
            **index.chunks[doc_index],
            "bm25_score": float(score),
        }
        for doc_index, score in bm25_hits
    ]
    fused = fuse_candidates(
        dense_candidates=dense_candidates,
        bm25_candidates=bm25_candidates,
        method=method,
        top_k=output_top_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
        rrf_k=rrf_k,
    )
    gold_articles = {str(article_id) for article_id in row.get("gold_articles", [])}
    metrics = evaluate_query(
        [str(candidate["article_id"]) for candidate in fused],
        gold_articles,
    )
    return {
        "qa_id": row.get("qa_id"),
        "qa_type": row.get("qa_type"),
        "question": row.get("question"),
        "gold_articles": sorted(gold_articles),
        "embedding_type": row.get("embedding_type", "dense"),
        "retriever": f"hybrid_{method}",
        "fusion": {
            "method": method,
            "dense_weight": dense_weight,
            "bm25_weight": bm25_weight,
            "rrf_k": rrf_k if method == "rrf" else None,
        },
        "top_k_dense": len(dense_candidates),
        "top_k_bm25": len(bm25_candidates),
        "top_k_output": output_top_k,
        "candidates": fused,
        "top_articles": [candidate["article_id"] for candidate in fused],
        "retrieval_metrics": metrics,
        **metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hybrid retrieval: existing dense candidates + BM25 over the same chunks."
    )
    parser.add_argument("--dense-input", default="src/embedding/output/per_query_structured.jsonl")
    parser.add_argument(
        "--chunks",
        default="src/chunking/output/vieonline_news_chunks_structured.jsonl",
    )
    parser.add_argument(
        "--output",
        default="src/retrieval/output/per_query_hybrid_structured_rrf.jsonl",
    )
    parser.add_argument(
        "--summary",
        default="src/retrieval/output/hybrid_structured_rrf_summary.json",
    )
    parser.add_argument("--fusion", choices=("rrf", "weighted"), default="rrf")
    parser.add_argument("--dense-top-k", type=int, default=10)
    parser.add_argument("--bm25-top-k", type=int, default=50)
    parser.add_argument("--output-top-k", type=int, default=10)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--bm25-weight", type=float, default=0.5)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--limit", type=int, default=None, help="Limit QA rows.")
    parser.add_argument(
        "--limit-chunks",
        type=int,
        default=None,
        help="Debug only: restrict the BM25 corpus.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dense_weight < 0 or args.bm25_weight < 0:
        raise SystemExit("Fusion weights must be non-negative")
    if math.isclose(args.dense_weight + args.bm25_weight, 0.0):
        raise SystemExit("At least one fusion weight must be positive")

    dense_path = Path(args.dense_input)
    chunk_path = Path(args.chunks)
    if not dense_path.exists():
        raise FileNotFoundError(f"Dense input not found: {dense_path}")
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunk_path}")

    rows = list(read_jsonl(dense_path))
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("Dense input contains no query rows")

    started = time.perf_counter()
    chunks = load_chunks(chunk_path, args.limit_chunks)
    print(f"Loaded {len(chunks)} chunks; building query-aware BM25 index...")
    index = BM25Index(
        chunks,
        query_vocabulary(rows),
        k1=args.bm25_k1,
        b=args.bm25_b,
    )
    index_seconds = time.perf_counter() - started

    outputs: List[Dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=1):
        outputs.append(
            build_output_row(
                row=row,
                index=index,
                method=args.fusion,
                dense_top_k=args.dense_top_k,
                bm25_top_k=args.bm25_top_k,
                output_top_k=args.output_top_k,
                dense_weight=args.dense_weight,
                bm25_weight=args.bm25_weight,
                rrf_k=args.rrf_k,
            )
        )
        if row_number % 25 == 0:
            print(f"Hybrid retrieval completed for {row_number}/{len(rows)} queries")

    output_path = Path(args.output)
    write_jsonl(output_path, outputs)
    metric_names = ("ndcg@10", "recall@5", "recall@10", "mrr@10", "hit@1", "hit@5")
    summary = {
        "retriever": f"hybrid_{args.fusion}",
        "num_queries": len(outputs),
        "num_chunks": len(chunks),
        "bm25_index_seconds": round(index_seconds, 3),
        "dense_input": str(dense_path),
        "chunks": str(chunk_path),
        "fusion": {
            "method": args.fusion,
            "dense_weight": args.dense_weight,
            "bm25_weight": args.bm25_weight,
            "rrf_k": args.rrf_k if args.fusion == "rrf" else None,
        },
        "metrics": {
            metric: round(mean(float(row.get(metric, 0.0)) for row in outputs), 6)
            for metric in metric_names
        },
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {len(outputs)} rows to {output_path}")


if __name__ == "__main__":
    main()
