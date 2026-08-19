"""Run BGE cross-encoder reranking over retrieval JSONL candidates."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rerank_rows(
    rows: list[dict[str, Any]],
    model_name: str,
    top_k: int,
    batch_size: int,
    max_length: int,
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    output: list[dict[str, Any]] = []
    for row in rows:
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        candidates = row.get("candidates") or row.get("retrieved_candidates") or []
        pairs = [(str(row.get("question", "")), str(item.get("text", ""))) for item in candidates]
        scores: list[float] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoded = tokenizer(
                [pair[0] for pair in batch],
                [pair[1] for pair in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits.reshape(-1).detach().cpu().tolist()
            scores.extend(float(score) for score in logits)

        ranked = []
        for candidate, score in zip(candidates, scores):
            ranked.append({**candidate, "original_rank": candidate.get("rank"), "rerank_score": score})
        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        ranked = [{**item, "rank": index + 1} for index, item in enumerate(ranked[:top_k])]
        if device == "cuda":
            torch.cuda.synchronize()
        output.append(
            {
                **row,
                "reranked_candidates": ranked,
                "reranker_model": model_name,
                "reranker_backend": "transformers_sequence_classification",
                "rerank_latency_ms": round((time.perf_counter() - started) * 1000, 4),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank retrieval candidates with a BGE reranker.")
    parser.add_argument("--input", type=Path, required=True, help="Input retrieval JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output reranked JSONL")
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    rows = rerank_rows(load_jsonl(args.input), args.model, args.top_k, args.batch_size, args.max_length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} queries to {args.output}")


if __name__ == "__main__":
    main()
