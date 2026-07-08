import argparse
import torch
import warnings
warnings.filterwarnings("ignore", message=".*flash_attn.*")
warnings.filterwarnings("ignore", message=".*flash-attn.*")
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:
    from transformers import AutoModelForSequenceClassification
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: transformers. Install with: pip install transformers"
    ) from exc

DEFAULT_INPUT = "src/embedding/output/per_query_structured.jsonl"
DEFAULT_OUTPUT = "src/re-ranker/output_Rerank/rerank_structure_jina_top5.jsonl"
DEFAULT_MODEL = "jinaai/jina-reranker-v2-base-multilingual"


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def unique_article_ids(article_ids: Sequence[str], k: int) -> List[str]:
    unique_ids: List[str] = []
    seen = set()
    for article_id in article_ids:
        if article_id in seen:
            continue
        seen.add(article_id)
        unique_ids.append(article_id)
        if len(unique_ids) == k:
            break
    return unique_ids


def hit_at_k(article_ids: Sequence[str], gold_articles: Sequence[str], k: int) -> float:
    gold_set = set(gold_articles)
    unique_ids = unique_article_ids(article_ids, k)
    return float(any(article_id in gold_set for article_id in unique_ids))


def recall_at_k(article_ids: Sequence[str], gold_articles: Sequence[str], k: int) -> float:
    gold_set = set(gold_articles)
    if not gold_set:
        return 0.0
    unique_ids = unique_article_ids(article_ids, k)
    found = set(unique_ids) & gold_set
    return len(found) / len(gold_set)


def mrr_at_k(article_ids: Sequence[str], gold_articles: Sequence[str], k: int) -> float:
    gold_set = set(gold_articles)
    unique_ids = unique_article_ids(article_ids, k)
    for index, article_id in enumerate(unique_ids, start=1):
        if article_id in gold_set:
            return 1.0 / index
    return 0.0


def ndcg_at_k(article_ids: Sequence[str], gold_articles: Sequence[str], k: int) -> float:
    gold_set = set(gold_articles)
    if not gold_set:
        return 0.0

    unique_ids = unique_article_ids(article_ids, k)
    relevances = [1.0 if article_id in gold_set else 0.0 for article_id in unique_ids]
    dcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances))

    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return min(dcg / idcg, 1.0) if idcg else 0.0


def compute_metrics(candidates: Sequence[Dict[str, Any]], gold_articles: Sequence[str], k: int) -> Dict[str, float]:
    article_ids = [str(candidate.get("article_id", "")) for candidate in candidates]
    return {
        "hit@1": hit_at_k(article_ids, gold_articles, 1),
        "hit@" + str(k): hit_at_k(article_ids, gold_articles, k),
        "recall@" + str(k): recall_at_k(article_ids, gold_articles, k),
        "mrr@" + str(k): mrr_at_k(article_ids, gold_articles, k),
        "ndcg@" + str(k): ndcg_at_k(article_ids, gold_articles, k),
    }


def batch_items(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def score_pairs(reranker: Any, pairs: List[List[str]], batch_size: int, max_length: int) -> List[float]:
    scores: List[float] = []
    for batch in batch_items(pairs, batch_size):
        batch_scores = reranker.compute_score(list(batch), max_length=max_length)
        if isinstance(batch_scores, float):
            scores.append(batch_scores)
        else:
            scores.extend(float(score) for score in batch_scores)
    return scores


def rerank_row(
    row: Dict[str, Any],
    reranker: Any,
    model_name: str,
    top_n: int,
    batch_size: int,
    max_length: int,
) -> Dict[str, Any]:
    question = row.get("question", "")
    candidates = row.get("candidates", [])
    gold_articles = [str(article_id) for article_id in row.get("gold_articles", [])]

    pairs = [[question, candidate.get("text", "")] for candidate in candidates]
    scores = score_pairs(reranker, pairs, batch_size, max_length) if pairs else []

    reranked_candidates = []
    for candidate, rerank_score in zip(candidates, scores):
        reranked_candidates.append(
            {
                "rank": 0,
                "original_rank": candidate.get("rank"),
                "chunk_index": candidate.get("chunk_index"),
                "chunk_id": candidate.get("chunk_id"),
                "article_id": candidate.get("article_id"),
                "retrieval_score": candidate.get("score"),
                "rerank_score": float(rerank_score),
                "text": candidate.get("text", ""),
            }
        )

    reranked_candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
    top_candidates = reranked_candidates[:top_n]
    for rank, candidate in enumerate(top_candidates, start=1):
        candidate["rank"] = rank

    metrics = compute_metrics(top_candidates, gold_articles, top_n)

    return {
        "qa_id": row.get("qa_id"),
        "qa_type": row.get("qa_type"),
        "question": question,
        "gold_articles": gold_articles,
        "embedding_type": "structured",
        "reranker": model_name,
        "top_k_input": len(candidates),
        "top_n_output": top_n,
        "reranked_candidates": top_candidates,
        "rerank_metrics": metrics,
        "original_retrieval_metrics": {
            key: row[key]
            for key in ("hit@1", "hit@5", "recall@5", "recall@10", "mrr@10", "ndcg@10")
            if key in row
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank structured embedding output with Jina reranker.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input per-query structured JSONL file.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output reranked JSONL file.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Jina reranker model name.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of reranked candidates to keep.")
    parser.add_argument("--batch-size", type=int, default=8, help="Pair scoring batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of queries to process for testing.")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum pair length for Jina scoring.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model to device: {device}")
    reranker = AutoModelForSequenceClassification.from_pretrained(args.model_name, trust_remote_code=True)
    reranker = reranker.to(device)
    rows = read_jsonl(input_path)

    outputs = []
    for index, row in enumerate(rows, start=1):
        if args.limit is not None and index > args.limit:
            break
        outputs.append(rerank_row(row, reranker, args.model_name, args.top_n, args.batch_size, args.max_length))
        if index % 10 == 0:
            print(f"Reranked {index} queries")

    write_jsonl(output_path, outputs)
    print(f"Done. Wrote {len(outputs)} rows to {output_path}")


if __name__ == "__main__":
    main()






