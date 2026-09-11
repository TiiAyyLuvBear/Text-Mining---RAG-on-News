"""Offline A/B benchmark for sentence compression on reranker JSONL output."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .context_compression import compress_context_by_sentence, default_token_count


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def benchmark_rows(
    rows: list[dict[str, Any]],
    thresholds: list[float],
    *,
    top_n_context: int = 5,
) -> list[dict[str, Any]]:
    results = []
    for threshold in thresholds:
        tokens_before = tokens_after = original_sentences = kept_sentences = 0
        mappings_total = mappings_preserved = 0
        started = time.perf_counter()
        for row in rows:
            contexts = [dict(item) for item in row.get("reranked_candidates", [])[:top_n_context]]
            for index, context in enumerate(contexts, start=1):
                context.setdefault("citation_rank", index)
            compressed, stats = compress_context_by_sentence(
                str(row.get("question") or ""),
                contexts,
                threshold=threshold,
                token_counter=default_token_count,
            )
            tokens_before += stats["tokens_before"]
            tokens_after += stats["tokens_after"]
            original_sentences += stats["original_sentence_count"]
            kept_sentences += stats["kept_sentence_count"]
            for before, after in zip(contexts, compressed):
                mappings_total += 1
                if all(
                    before.get(key) == after.get(key)
                    for key in ("article_id", "chunk_id", "citation_rank")
                ):
                    mappings_preserved += 1
        elapsed_ms = (time.perf_counter() - started) * 1000
        results.append({
            "threshold": threshold,
            "queries": len(rows),
            "top_n_context": top_n_context,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "token_reduction_ratio": round(1 - tokens_after / tokens_before, 4) if tokens_before else 0.0,
            "original_sentence_count": original_sentences,
            "kept_sentence_count": kept_sentences,
            "sentence_reduction_ratio": round(1 - kept_sentences / original_sentences, 4) if original_sentences else 0.0,
            "source_mapping_preservation": round(mappings_preserved / mappings_total, 4) if mappings_total else 1.0,
            "compression_latency_ms_total": round(elapsed_ms, 3),
            "compression_latency_ms_mean": round(elapsed_ms / len(rows), 3) if rows else 0.0,
        })
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.15, 0.25, 0.35])
    parser.add_argument("--top-n-context", type=int, default=5)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(args.input))
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
    report = {
        "input": str(args.input),
        "token_counter": "regex fallback; model tokenizer not loaded",
        "generation_metrics": "not run; requires generator and upstream contracts",
        "results": benchmark_rows(rows, args.thresholds, top_n_context=args.top_n_context),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
