from __future__ import annotations

import argparse
import json
from pathlib import Path

from .strategies import ChunkingConfig, chunk_records, read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create chunk JSONL files for RAG experiments.")
    parser.add_argument(
        "--input",
        default="data/processed/vieonline_news_clean.jsonl",
        help="Input preprocessed article JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/chunks",
        help="Directory where chunk JSONL and summary files are written.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["fixed", "sentence", "paragraph", "hybrid"],
        choices=["fixed", "sentence", "paragraph", "hybrid", "llamaindex"],
        help="Chunking strategies to run.",
    )
    parser.add_argument("--chunk-size", type=int, default=450)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--sentence-overlap", type=int, default=2)
    parser.add_argument("--min-chunk-tokens", type=int, default=80)
    parser.add_argument("--small-article-chars", type=int, default=1000)
    parser.add_argument("--long-article-chars", type=int, default=20000)
    parser.add_argument("--max-chunks-per-article", type=int, default=None)
    parser.add_argument(
        "--no-title-injection",
        action="store_true",
        help="Do not prepend title/description/category to each chunk text.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of input articles to process for quick experiments.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {}
    for strategy in args.strategies:
        config = ChunkingConfig(
            strategy=strategy,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            sentence_overlap=args.sentence_overlap,
            min_chunk_tokens=args.min_chunk_tokens,
            small_article_chars=args.small_article_chars,
            long_article_chars=args.long_article_chars,
            max_chunks_per_article=args.max_chunks_per_article,
            inject_title_description=not args.no_title_injection,
        )
        output_path = output_dir / f"vieonline_news_chunks_{strategy}.jsonl"
        stats = chunk_records(read_jsonl(args.input, limit=args.limit), output_path, config)
        summary[strategy] = {
            "output_path": str(output_path),
            "config": config.__dict__,
            "stats": stats.to_dict(),
        }
        print(json.dumps({strategy: summary[strategy]}, ensure_ascii=False, indent=2))

    summary_path = output_dir / "chunking_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
