"""Create answer-to-chunk *silver* evidence labels for VietOnline QA.

The source QA file only identifies an article.  This tool restricts candidates
to those source articles and selects one evidence chunk per article from its
gold answer.  Unanswerable rows deliberately receive no gold chunks.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .qrels import article_ids_from_qa
from .tokenize import tokenize_vietnamese, tokenizer_name


DEFAULT_QA_PATH = Path("Dataset/QA_Claude/QA_output.jsonl")
DEFAULT_CHUNKS_PATH = Path("src/chunking/output/vieonline_news_chunks_token.jsonl")
DEFAULT_OUTPUT_PATH = Path("Dataset/QA_Claude/QA_output_gold_chunks.jsonl")
DEFAULT_REPORT_PATH = Path("Dataset/QA_Claude/QA_output_gold_chunks_report.json")
GENERATED_METADATA_FIELDS = (
    "gold_article_ids",
    "gold_label_type",
    "gold_label_version",
    "gold_chunk_ids",
    "gold_evidence",
    "gold_articles_without_chunk",
    "gold_label_status",
)

# Remove frequent function words so an overlap means evidence, not Vietnamese
# grammar shared by every chunk.
STOPWORDS = frozenset(
    "anh a anh ấy bà bạn bị bởi các cái cho chẳng chỉ có của cùng cũng đã đang để điều đều đây đó được đến gì khi không là lại mà một mọi như nữa ở phải rồi sẽ sự theo thì trong từ về và với vì vẫn vẫn".split()
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def evidence_terms(text: str) -> set[str]:
    return {
        token
        for token in tokenize_vietnamese(text)
        if len(token) > 1 and token not in STOPWORDS
    }


def lexical_evidence_score(answer: str, chunk: str) -> tuple[float, float, float]:
    """Return weighted score, answer coverage, and token F1."""
    answer_terms = evidence_terms(answer)
    chunk_terms = evidence_terms(chunk)
    if not answer_terms or not chunk_terms:
        return 0.0, 0.0, 0.0
    overlap = len(answer_terms & chunk_terms)
    coverage = overlap / len(answer_terms)
    precision = overlap / len(chunk_terms)
    f1 = 2 * precision * coverage / (precision + coverage) if precision + coverage else 0.0
    return 0.7 * coverage + 0.3 * f1, coverage, f1


def chunks_for_source_articles(chunks_path: Path, article_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with chunks_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chunk = json.loads(line)
            article_id = str(chunk.get("article_id", ""))
            if article_id in article_ids:
                indexed[article_id].append(chunk)
    return indexed


def select_gold_chunks(
    qa_rows: list[dict[str, Any]],
    chunks_by_article: dict[str, list[dict[str, Any]]],
    *,
    min_score: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    result: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for qa in qa_rows:
        row = dict(qa)
        possible = bool(row.get("is_possible", True))
        source_article_ids = article_ids_from_qa(row)
        row["gold_article_ids"] = source_article_ids if possible else []
        row["gold_label_type"] = "silver_answer_chunk_alignment" if possible else "unanswerable_no_evidence"
        row["gold_label_version"] = "v1"

        if not possible:
            row["gold_id"] = []
            row["gold_chunk_ids"] = []
            row["gold_evidence"] = []
            row["gold_label_status"] = "unanswerable"
            status_counts["unanswerable"] += 1
            result.append(row)
            continue

        answers = row.get("answers", [])
        answer = " ".join(str(item).strip() for item in answers if str(item).strip()) if isinstance(answers, list) else str(answers).strip()
        if not answer:
            row["gold_id"] = []
            row["gold_chunk_ids"] = []
            row["gold_evidence"] = []
            row["gold_articles_without_chunk"] = source_article_ids
            row["gold_label_status"] = "needs_review_missing_gold_answer"
            status_counts["needs_review_missing_gold_answer"] += 1
            result.append(row)
            continue
        evidence: list[dict[str, Any]] = []
        missing_articles: list[str] = []
        for article_id in source_article_ids:
            candidates = chunks_by_article.get(article_id, [])
            scored = []
            for chunk in candidates:
                chunk_text = str(chunk.get("chunk_text") or chunk.get("text") or "")
                score, coverage, f1 = lexical_evidence_score(answer, chunk_text)
                scored.append((score, coverage, f1, chunk))
            if not scored:
                missing_articles.append(article_id)
                continue
            score, coverage, f1, chunk = max(scored, key=lambda item: (item[0], item[1], str(item[3].get("chunk_id", ""))))
            if score < min_score:
                missing_articles.append(article_id)
                continue
            evidence.append(
                {
                    "article_id": article_id,
                    "chunk_id": str(chunk["chunk_id"]),
                    "score": round(score, 6),
                    "answer_term_coverage": round(coverage, 6),
                    "token_f1": round(f1, 6),
                    "matching": "lexical_answer_chunk",
                }
            )

        row["gold_id"] = [item["chunk_id"] for item in evidence]
        row["gold_chunk_ids"] = row["gold_id"]
        row["gold_evidence"] = evidence
        row["gold_articles_without_chunk"] = missing_articles
        if not source_article_ids:
            status = "needs_review_no_source_article"
        elif missing_articles:
            status = "needs_review_partial_alignment"
        else:
            status = "silver_ready"
        row["gold_label_status"] = status
        status_counts[status] += 1
        result.append(row)
    # Dataset schema stays identical to source QA, plus only ``gold_id``.
    # Audit metadata belongs in the separate report, not each QA record.
    for row in result:
        for field in GENERATED_METADATA_FIELDS:
            row.pop(field, None)
    return result, status_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Add silver gold_chunk_ids to VietOnline QA JSONL.")
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA_PATH)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--min-score", type=float, default=0.12)
    args = parser.parse_args()

    qa_rows = read_jsonl(args.qa)
    source_ids = {
        article_id
        for qa in qa_rows
        if bool(qa.get("is_possible", True))
        for article_id in article_ids_from_qa(qa)
    }
    chunks_by_article = chunks_for_source_articles(args.chunks, source_ids)
    labelled, statuses = select_gold_chunks(qa_rows, chunks_by_article, min_score=args.min_score)
    write_jsonl(args.output, labelled)

    possible_rows = [row for row in labelled if bool(row.get("is_possible", True))]
    report = {
        "qa_path": str(args.qa),
        "chunks_path": str(args.chunks),
        "output_path": str(args.output),
        "chunk_strategy": "token",
        "tokenizer": tokenizer_name(),
        "matching": "answer-to-chunk lexical coverage + token F1; one best chunk per source article",
        "min_score": args.min_score,
        "qa_total": len(labelled),
        "answerable": len(possible_rows),
        "unanswerable": len(labelled) - len(possible_rows),
        "answerable_with_all_source_articles_aligned": statuses["silver_ready"],
        "answerable_needing_review": len(possible_rows) - statuses["silver_ready"],
        "unanswerable_with_empty_gold_chunks": sum(not row["gold_id"] for row in labelled if not bool(row.get("is_possible", True))),
        "status_counts": dict(statuses),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
