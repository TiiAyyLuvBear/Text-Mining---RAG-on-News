from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_chunks_by_article(path: str | Path) -> dict[str, list[dict[str, object]]]:
    by_article: dict[str, list[dict[str, object]]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_article[str(row["article_id"])].append(row)
    return dict(by_article)


def infer_strategy(path: str | Path, rows: dict[str, list[dict[str, object]]]) -> str:
    for chunks in rows.values():
        if chunks:
            return str(chunks[0].get("strategy") or Path(path).stem)
    return Path(path).stem


def build_comparison_report(input_paths: list[str | Path], *, sample_articles: int) -> str:
    datasets = []
    for path in input_paths:
        rows = read_chunks_by_article(path)
        datasets.append((infer_strategy(path, rows), Path(path), rows))

    common_articles = set(datasets[0][2])
    for _, _, rows in datasets[1:]:
        common_articles &= set(rows)
    selected_articles = sorted(common_articles)[:sample_articles]

    lines = [
        "# Embedding Strategy Samples",
        "",
        "Report này so sánh text/metadata được đưa vào embedding giữa các chunking strategy cho cùng article.",
        "",
    ]
    if not selected_articles:
        lines.append("Không tìm thấy article_id chung giữa các input.")
        return "\n".join(lines) + "\n"

    for article_id in selected_articles:
        lines.extend([f"## Article `{article_id}`", ""])
        for strategy, path, rows in datasets:
            chunks = rows[article_id]
            lines.extend(
                [
                    f"### `{strategy}`",
                    "",
                    f"- Source: `{path}`",
                    f"- Num chunks: {len(chunks)}",
                    "",
                    "| Chunk | Implementation | Structure | Tokens | Preview |",
                    "| ---: | --- | --- | ---: | --- |",
                ]
            )
            for row in chunks[:5]:
                metadata = dict(row.get("metadata") or {})
                preview = _preview(str(row.get("text") or ""), 220).replace("|", "\\|")
                lines.append(
                    "| {chunk} | `{implementation}` | `{structure}` | {tokens} | {preview} |".format(
                        chunk=metadata.get("chunk_index", ""),
                        implementation=metadata.get("implementation", ""),
                        structure=metadata.get("structure", ""),
                        tokens=metadata.get("token_count", ""),
                        preview=preview,
                    )
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def _preview(text: str, limit: int) -> str:
    value = " ".join(text.split())
    return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare embedding inputs produced by different chunking strategies.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Chunk JSONL files to compare.")
    parser.add_argument("--sample-articles", type=int, default=3)
    parser.add_argument("--output", default="src/embed/output/embedding_strategy_samples.md")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_comparison_report(args.inputs, sample_articles=args.sample_articles)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



