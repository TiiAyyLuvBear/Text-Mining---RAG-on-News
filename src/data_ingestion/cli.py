from __future__ import annotations

import argparse
import json
from pathlib import Path

from .preprocess import PreprocessConfig, process_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess VieOnlineNews CSV into JSONL for RAG ingestion."
    )
    parser.add_argument(
        "--input",
        default="Dataset/VieOnlineNews.csv",
        help="Input CSV path. Default: Dataset/VieOnlineNews.csv",
    )
    parser.add_argument(
        "--output",
        default="data/processed/vieonline_news_clean.jsonl",
        help="Output cleaned JSONL path.",
    )
    parser.add_argument(
        "--review-output",
        default="data/processed/vieonline_news_human_review.jsonl",
        help="Optional JSONL with raw/clean previews for human review. Use '' to disable.",
    )
    parser.add_argument(
        "--review-limit",
        type=int,
        default=200,
        help="Maximum rows written to the human review JSONL.",
    )
    parser.add_argument(
        "--min-content-chars",
        type=int,
        default=300,
        help="Flag content shorter than this as short_content.",
    )
    parser.add_argument(
        "--long-content-chars",
        type=int,
        default=20_000,
        help="Flag content longer than this as long_content.",
    )
    parser.add_argument(
        "--strip-urls",
        action="store_true",
        help="Remove URLs from text instead of only flagging them.",
    )
    parser.add_argument(
        "--content-only",
        action="store_true",
        help="Use cleaned content as text field instead of title/description/category/content template.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = PreprocessConfig(
        min_content_chars=args.min_content_chars,
        long_content_chars=args.long_content_chars,
        strip_urls=args.strip_urls,
        include_embedding_text=not args.content_only,
    )
    review_output = args.review_output or None
    stats = process_csv(
        Path(args.input),
        Path(args.output),
        review_output_path=Path(review_output) if review_output else None,
        review_limit=args.review_limit,
        config=config,
    )
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
