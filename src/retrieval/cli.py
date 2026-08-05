from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone

from .bm25 import build_bm25_index, load_bm25_index, search_bm25
from .evaluate import evaluate_retrievers, read_per_query_csv, summarize_per_query, write_evaluation
from .hybrid import reciprocal_rank_fusion
from .qrels import build_article_qrels, build_gold_chunk_qrels, build_weak_qrels, write_weak_qrels


DEFAULT_MODEL = "intfloat/multilingual-e5-large"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and evaluate BM25, dense Qdrant, and hybrid retrieval.")
    commands = parser.add_subparsers(dest="command", required=True)
    qrels = commands.add_parser("qrels"); qrels_sub = qrels.add_subparsers(dest="action", required=True); build = qrels_sub.add_parser("build")
    build.add_argument("--qa", required=True); build.add_argument("--chunks"); build.add_argument("--output", required=True); build.add_argument("--label-level", choices=("article", "chunk"), default="article"); build.add_argument("--use-gold-ids", action="store_true", help="For chunk labels, read gold_id/gold_chunk_ids already present in QA; no --chunks needed."); build.add_argument("--include-unanswerable", action="store_true", help="For article labels, include unanswerable QA as source-document retrieval cases."); build.add_argument("--semantic-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"); build.add_argument("--skip-qa-ids-from", help="Existing per_query.csv; build qrels only for QA IDs absent from it."); build.add_argument("--no-semantic", action="store_true"); build.add_argument("--no-progress", action="store_true")
    bm25 = commands.add_parser("bm25"); bm25_sub = bm25.add_subparsers(dest="action", required=True); build = bm25_sub.add_parser("build"); build.add_argument("--chunks", required=True); build.add_argument("--index-dir", required=True); build.add_argument("--no-progress", action="store_true")
    dense = commands.add_parser("dense"); dense_sub = dense.add_subparsers(dest="action", required=True); build = dense_sub.add_parser("build"); build.add_argument("--chunks", required=True); build.add_argument("--qdrant-path", default="data/indexes/qdrant"); build.add_argument("--collection", required=True); build.add_argument("--model", default=DEFAULT_MODEL); build.add_argument("--batch-size", type=int, default=16); build.add_argument("--rebuild", action="store_true")
    search = commands.add_parser("search"); search.add_argument("--method", choices=("bm25", "dense", "hybrid"), required=True); search.add_argument("--question", required=True); search.add_argument("--top-k", type=int, default=10); search.add_argument("--bm25-index-dir"); search.add_argument("--qdrant-path", default="data/indexes/qdrant"); search.add_argument("--collection"); search.add_argument("--model", default=DEFAULT_MODEL); search.add_argument("--rrf-k", type=int, default=60)
    evaluate = commands.add_parser("evaluate"); evaluate.add_argument("--qrels", required=True); evaluate.add_argument("--methods", nargs="+", choices=("bm25", "dense", "hybrid"), default=["bm25", "dense", "hybrid"]); evaluate.add_argument("--bm25-index-dir", required=True); evaluate.add_argument("--qdrant-path", default="data/indexes/qdrant"); evaluate.add_argument("--collection", required=True); evaluate.add_argument("--model", default=DEFAULT_MODEL); evaluate.add_argument("--candidate-k", type=int, default=50); evaluate.add_argument("--rrf-k", type=int, default=60); evaluate.add_argument("--output-dir", required=True); evaluate.add_argument("--no-progress", action="store_true")
    summarize = commands.add_parser("summarize"); summarize.add_argument("--per-query", nargs="+", required=True, help="One or more per_query.csv files to merge."); summarize.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "qrels":
        existing_ids: set[str] = set()
        if args.skip_qa_ids_from:
            with open(args.skip_qa_ids_from, encoding="utf-8", newline="") as handle:
                existing_ids = {str(row["qa_id"]) for row in csv.DictReader(handle)}
        if args.label_level == "article":
            qrels = build_article_qrels(args.qa, skip_qa_ids=existing_ids, include_unanswerable=args.include_unanswerable)
        elif args.use_gold_ids:
            qrels = build_gold_chunk_qrels(args.qa, skip_qa_ids=existing_ids)
        else:
            if not args.chunks:
                raise ValueError("--chunks is required when --label-level chunk.")
            from src.embed.embed_chunks import load_sentence_transformer

            encoder = None if args.no_semantic else load_sentence_transformer(args.semantic_model)
            qrels = build_weak_qrels(args.qa, args.chunks, semantic_encoder=encoder, show_progress=not args.no_progress, skip_qa_ids=existing_ids)
        write_weak_qrels(args.output, qrels); print(json.dumps({"output": args.output, "qrels": len(qrels)}, ensure_ascii=False)); return
    if args.command == "bm25": print(json.dumps(build_bm25_index(args.chunks, args.index_dir, show_progress=not args.no_progress), ensure_ascii=False)); return
    if args.command == "dense":
        from .dense_qdrant import build_qdrant_index

        print(json.dumps(build_qdrant_index(args.chunks, args.qdrant_path, args.collection, model_name=args.model, batch_size=args.batch_size, rebuild=args.rebuild), ensure_ascii=False)); return
    bm25_data = load_bm25_index(args.bm25_index_dir) if getattr(args, "bm25_index_dir", None) else None
    bm25_searcher = (lambda query, k: search_bm25(bm25_data, query, k)) if bm25_data is not None else None
    if getattr(args, "collection", None):
        from .dense_qdrant import search_qdrant

        dense_searcher = lambda query, k: search_qdrant(args.qdrant_path, args.collection, query, top_k=k, model_name=args.model)
    else:
        dense_searcher = None
    if args.command == "search":
        if args.method == "bm25": results = bm25_searcher(args.question, args.top_k)
        elif args.method == "dense": results = dense_searcher(args.question, args.top_k)
        else: results = reciprocal_rank_fusion([bm25_searcher(args.question, max(50, args.top_k)), dense_searcher(args.question, max(50, args.top_k))], rrf_k=args.rrf_k, top_k=args.top_k)
        print(json.dumps({"question": args.question, "results": results}, ensure_ascii=False, indent=2)); return
    if args.command == "summarize":
        per_query = read_per_query_csv(args.per_query)
        summary = summarize_per_query(per_query)
        output = write_evaluation(args.output_dir, summary, per_query, {"created_at": datetime.now(timezone.utc).isoformat(), "per_query_inputs": args.per_query, "qrels_are_weak_labels": True})
        print(json.dumps({"summary": summary, "outputs": output}, ensure_ascii=False, indent=2)); return
    methods = {}
    if "bm25" in args.methods: methods["bm25"] = bm25_searcher
    if "dense" in args.methods: methods["dense"] = dense_searcher
    if "hybrid" in args.methods: methods["hybrid"] = lambda query, k: reciprocal_rank_fusion([bm25_searcher(query, max(args.candidate_k, k)), dense_searcher(query, max(args.candidate_k, k))], rrf_k=args.rrf_k, top_k=k)
    summary, per_query = evaluate_retrievers(args.qrels, methods, candidate_k=args.candidate_k, show_progress=not args.no_progress)
    output = write_evaluation(args.output_dir, summary, per_query, {"created_at": datetime.now(timezone.utc).isoformat(), "qrels": args.qrels, "methods": args.methods, "candidate_k": args.candidate_k, "rrf_k": args.rrf_k, "model": args.model, "qrels_are_weak_labels": True})
    print(json.dumps({"summary": summary, "outputs": output}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
