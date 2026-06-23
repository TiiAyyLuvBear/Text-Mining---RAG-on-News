from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .strategies import DEFAULT_STRATEGIES, ChunkingConfig, chunk_records, read_jsonl
except ImportError:  # Allows `python src\chunking\cli.py ...`
    from strategies import DEFAULT_STRATEGIES, ChunkingConfig, chunk_records, read_jsonl

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create chunk JSONL files for RAG experiments.")
    parser.add_argument(
        "--input",
        default="data/processed/vieonline_news_clean.jsonl",
        help="Input preprocessed article JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default="src/chunking/output",
        help="Directory where chunk JSONL and summary files are written.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(DEFAULT_STRATEGIES),
        choices=list(DEFAULT_STRATEGIES),
        help="Chunking strategies to run.",
    )
    parser.add_argument("--chunk-size", type=int, default=450)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--min-chunk-tokens", type=int, default=80)
    parser.add_argument("--small-article-chars", type=int, default=1000)
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
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser


def count_jsonl_records(path: str | Path, *, limit: int | None = None) -> int:
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            count += 1
            if limit is not None and count >= limit:
                return limit
    return count


def progress_records(records, *, total: int, strategy: str, enabled: bool):
    if not enabled or tqdm is None:
        return records
    return tqdm(
        records,
        total=total,
        desc=f"Chunking [{strategy}]",
        unit="article",
        dynamic_ncols=True,
    )


def build_markdown_report(summary: dict[str, object]) -> str:
    lines = [
        "# Chunking Experiment Report",
        "",
        "| Strategy | Articles | Chunks | Avg chunks/article | Avg tokens/chunk | Time (s) | Chunks/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy, payload in summary.items():
        stats = payload["stats"]
        lines.append(
            "| {strategy} | {articles} | {chunks} | {avg_chunks} | {avg_tokens} | {elapsed} | {cps} |".format(
                strategy=strategy,
                articles=stats["articles_read"],
                chunks=stats["chunks_written"],
                avg_chunks=stats["avg_chunks_per_article"],
                avg_tokens=stats["avg_chunk_tokens"],
                elapsed=stats["elapsed_seconds"],
                cps=stats["chunks_per_second"],
            )
        )

    lines.extend(
        [
            "",
            "## Quick Analysis",
            "",
            "- `token` is the baseline: fixed token windows with token overlap.",
            "- `langchain_recursive` preserves larger text boundaries first, then falls back to smaller separators.",
            "- `llamaindex` uses `SentenceSplitter` when installed, with an internal sentence-window fallback otherwise.",
            "- `structured` respects article paragraph structure before applying sentence/token windows.",
            "",
            "Use `chunking_summary.json` for exact metrics and each `vieonline_news_chunks_<strategy>.jsonl` file for per-chunk inspection.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {}
    total_records = count_jsonl_records(args.input, limit=args.limit)
    progress_enabled = not args.no_progress
    if progress_enabled and tqdm is None:
        print("tqdm is not installed; progress bars are disabled.")

    print(
        f"Starting chunking: {total_records} article(s), "
        f"{len(args.strategies)} strategy(ies), output_dir={output_dir}"
    )
    for strategy in args.strategies:
        config = ChunkingConfig(
            strategy=strategy,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            min_chunk_tokens=args.min_chunk_tokens,
            small_article_chars=args.small_article_chars,
            max_chunks_per_article=args.max_chunks_per_article,
            inject_title_description=not args.no_title_injection,
        )
        output_path = output_dir / f"vieonline_news_chunks_{strategy}.jsonl"
        records = progress_records(
            read_jsonl(args.input, limit=args.limit),
            total=total_records,
            strategy=strategy,
            enabled=progress_enabled,
        )
        stats = chunk_records(records, output_path, config)
        summary[strategy] = {
            "output_path": str(output_path),
            "config": config.__dict__,
            "stats": stats.to_dict(),
        }
        print(
            f"Completed {strategy}: "
            f"{stats.articles_read} article(s), {stats.chunks_written} chunk(s), "
            f"{stats.elapsed_seconds}s -> {output_path}"
        )
        print(json.dumps({strategy: summary[strategy]}, ensure_ascii=False, indent=2))

    summary_path = output_dir / "chunking_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = output_dir / "chunking_report.md"
    report_path.write_text(build_markdown_report(summary), encoding="utf-8")
    total_chunks = sum(payload["stats"]["chunks_written"] for payload in summary.values())
    print(
        f"Chunking complete: {total_chunks} chunk(s) written across {len(summary)} strategy(ies). "
        f"Summary: {summary_path}. Report: {report_path}."
    )
    print(json.dumps({"summary_path": str(summary_path), "report_path": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
