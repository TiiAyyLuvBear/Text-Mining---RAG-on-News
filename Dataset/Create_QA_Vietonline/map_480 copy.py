"""Map existing QA answers to source chunk IDs without calling an LLM.

Answerable single-article questions receive up to a few best matching chunks.
Answerable cross-article questions receive at least one chunk from every source
article. Unanswerable questions receive an empty ``gold_id`` list.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_QA_PATH = BASE_DIR / "QA_output_new_480_update.jsonl"
DEFAULT_CHUNK_PATH = BASE_DIR / "vieonline_news_chunks_token.jsonl"

BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_TOP_K_SINGLE = 3
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

# Stop words reduce the influence of generic Vietnamese function words. Numbers
# and named entities remain in the query because they are valuable for QA match.
STOP_WORDS = {
    "ai", "anh", "bà", "bài", "bạn", "bị", "bởi", "cả", "các", "cái",
    "cần", "chỉ", "cho", "có", "còn", "của", "cũng", "đã", "đang", "đây",
    "để", "đến", "được", "do", "đó", "gì", "hay", "hơn", "khi", "không",
    "là", "lại", "mà", "một", "nào", "này", "nên", "những", "như", "ở",
    "ra", "rằng", "sau", "sẽ", "theo", "thì", "trên", "trong", "từ", "và",
    "vào", "về", "vì", "với",
}


def normalize_article_id(value) -> str:
    """Normalize numeric/string IDs to the string format used by chunks."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def normalize_text(value) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(TOKEN_RE.findall(text))


def tokenize(value) -> list[str]:
    return [
        token
        for token in TOKEN_RE.findall(normalize_text(value))
        if (len(token) > 1 or token.isdigit()) and token not in STOP_WORDS
    ]


def source_article_ids(record: dict) -> list[str]:
    """Return source IDs in stable order, preferring article_id."""
    raw_ids = record.get("article_id")
    if not isinstance(raw_ids, list):
        raw_ids = [raw_ids]
    if not any(value is not None for value in raw_ids):
        fallback = record.get("source_article_ids", [])
        raw_ids = fallback if isinstance(fallback, list) else [fallback]

    result = []
    seen = set()
    for value in raw_ids:
        article_id = normalize_article_id(value)
        if article_id and article_id not in seen:
            seen.add(article_id)
            result.append(article_id)
    return result


def load_qa(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def load_relevant_chunks(
    path: Path,
    relevant_article_ids: set[str],
) -> tuple[dict[str, list[dict]], int]:
    """Stream the large chunk file and retain only articles used by the QA."""
    chunks_by_article: dict[str, list[dict]] = defaultdict(list)
    malformed = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue

            article_id = normalize_article_id(
                chunk.get("article_id") or chunk.get("metadata", {}).get("article_id")
            )
            if article_id not in relevant_article_ids:
                continue

            chunk_id = str(chunk.get("chunk_id", "")).strip()
            chunk_text = chunk.get("chunk_text") or chunk.get("text") or ""
            if not chunk_id or not chunk_text:
                malformed += 1
                continue

            tokens = tokenize(chunk_text)
            chunks_by_article[article_id].append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "normalized_text": normalize_text(chunk_text),
                "tokens": tokens,
                "token_counts": Counter(tokens),
                "token_set": set(tokens),
            })

    for chunks in chunks_by_article.values():
        chunks.sort(key=lambda chunk: chunk["chunk_id"])
    return dict(chunks_by_article), malformed


def bm25_scores(chunks: list[dict], query_tokens: list[str]) -> list[float]:
    """Compute BM25 scores inside one article's small chunk collection."""
    if not chunks or not query_tokens:
        return [0.0] * len(chunks)

    query_counts = Counter(query_tokens)
    document_count = len(chunks)
    avg_length = sum(len(chunk["tokens"]) for chunk in chunks) / document_count
    avg_length = max(avg_length, 1.0)
    document_frequency = {
        token: sum(token in chunk["token_set"] for chunk in chunks)
        for token in query_counts
    }

    scores = []
    for chunk in chunks:
        document_length = max(len(chunk["tokens"]), 1)
        score = 0.0
        for token, query_frequency in query_counts.items():
            term_frequency = chunk["token_counts"].get(token, 0)
            if not term_frequency:
                continue
            frequency = document_frequency[token]
            inverse_document_frequency = math.log(
                1.0 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            denominator = term_frequency + BM25_K1 * (
                1.0 - BM25_B + BM25_B * document_length / avg_length
            )
            score += (
                inverse_document_frequency
                * term_frequency
                * (BM25_K1 + 1.0)
                / denominator
                * min(query_frequency, 2)
            )
        scores.append(score)
    return scores


def rank_chunks(chunks: list[dict], answer_texts: Iterable[str]) -> list[dict]:
    """Rank chunks using BM25 plus answer-token and exact-phrase coverage."""
    answers = [str(answer).strip() for answer in answer_texts if str(answer).strip()]
    query_tokens = tokenize(" ".join(answers))
    query_token_set = set(query_tokens)
    scores = bm25_scores(chunks, query_tokens)

    ranked = []
    for chunk, bm25_score in zip(chunks, scores):
        overlap = query_token_set & chunk["token_set"]
        coverage = len(overlap) / max(len(query_token_set), 1)
        exact_matches = sum(
            len(normalize_text(answer)) >= 8
            and normalize_text(answer) in chunk["normalized_text"]
            for answer in answers
        )
        ranked.append({
            "chunk": chunk,
            "score": bm25_score + 5.0 * coverage + 10.0 * exact_matches,
            "coverage": coverage,
            "covered_tokens": overlap,
        })

    ranked.sort(key=lambda item: (-item["score"], item["chunk"]["chunk_id"]))
    return ranked


def select_single_gold_ids(
    chunks: list[dict],
    answers: list[str],
    top_k: int,
) -> list[str]:
    """Choose best chunks per answer, then add chunks with useful new coverage."""
    if not chunks or not answers:
        return []

    selected: list[dict] = []
    selected_ids = set()

    # Separate ranking prevents one answer in a multi-answer record from hiding
    # the chunk that supports another answer.
    for answer in answers:
        ranked = rank_chunks(chunks, [answer])
        if ranked:
            best = ranked[0]["chunk"]
            if best["chunk_id"] not in selected_ids:
                selected.append(best)
                selected_ids.add(best["chunk_id"])
        if len(selected) >= top_k:
            break

    combined_ranking = rank_chunks(chunks, answers)
    if not selected and combined_ranking:
        selected.append(combined_ranking[0]["chunk"])
        selected_ids.add(combined_ranking[0]["chunk"]["chunk_id"])

    query_tokens = set(tokenize(" ".join(answers)))
    covered_tokens = set().union(*(chunk["token_set"] for chunk in selected)) if selected else set()

    # Add another chunk only when it contributes meaningful answer vocabulary.
    while len(selected) < top_k and query_tokens:
        missing_tokens = query_tokens - covered_tokens
        if len(missing_tokens) / len(query_tokens) <= 0.25:
            break
        candidates = [
            item for item in combined_ranking
            if item["chunk"]["chunk_id"] not in selected_ids
        ]
        if not candidates:
            break
        best = max(
            candidates,
            key=lambda item: (
                len(item["chunk"]["token_set"] & missing_tokens),
                item["score"],
            ),
        )
        new_tokens = best["chunk"]["token_set"] & missing_tokens
        if len(new_tokens) < 2:
            break
        selected.append(best["chunk"])
        selected_ids.add(best["chunk"]["chunk_id"])
        covered_tokens.update(best["chunk"]["token_set"])

    return [chunk["chunk_id"] for chunk in selected]


def select_cross_gold_ids(
    chunks_by_article: dict[str, list[dict]],
    article_ids: list[str],
    answers: list[str],
) -> tuple[list[str], list[str]]:
    """Select the best chunk from every source article in a cross QA."""
    gold_ids = []
    missing_article_ids = []
    for article_id in article_ids:
        chunks = chunks_by_article.get(article_id, [])
        if not chunks:
            missing_article_ids.append(article_id)
            continue
        ranked = rank_chunks(chunks, answers)
        if not ranked:
            missing_article_ids.append(article_id)
            continue
        gold_ids.append(ranked[0]["chunk"]["chunk_id"])
    return gold_ids, missing_article_ids


def validated_source_chunk_hints(
    record: dict,
    chunks_by_article: dict[str, list[dict]],
) -> list[str]:
    """Accept generator hints only when they exist and cover every source."""
    raw_hints = record.get("source_chunk_ids", [])
    if not isinstance(raw_hints, list):
        raw_hints = [raw_hints]
    hints = []
    for value in raw_hints:
        chunk_id = str(value).strip()
        if chunk_id and chunk_id not in hints:
            hints.append(chunk_id)

    article_ids = source_article_ids(record)
    available_by_article = {
        article_id: {
            chunk["chunk_id"] for chunk in chunks_by_article.get(article_id, [])
        }
        for article_id in article_ids
    }
    valid_hints = [
        chunk_id
        for chunk_id in hints
        if any(chunk_id in available for available in available_by_article.values())
    ]
    if not article_ids:
        return []
    if not all(
        any(chunk_id in available for chunk_id in valid_hints)
        for available in available_by_article.values()
    ):
        return []
    return valid_hints


def add_gold_ids(
    records: list[dict],
    chunks_by_article: dict[str, list[dict]],
    top_k_single: int,
) -> dict:
    stats = Counter()
    missing_sources = set()

    for record in records:
        answers = [
            str(answer).strip()
            for answer in record.get("answers", [])
            if str(answer).strip()
        ]
        if record.get("is_possible") is False or not answers:
            record["gold_id"] = []
            stats["no_answer"] += 1
            continue

        article_ids = source_article_ids(record)
        is_cross = len(article_ids) > 1
        hinted_gold_ids = validated_source_chunk_hints(record, chunks_by_article)
        if is_cross:
            if hinted_gold_ids:
                gold_ids, missing = hinted_gold_ids, []
                stats["hinted_records"] += 1
            else:
                gold_ids, missing = select_cross_gold_ids(
                    chunks_by_article, article_ids, answers
                )
            record["gold_id"] = gold_ids
            missing_sources.update(missing)
            stats["cross_answerable"] += 1
            stats["cross_gold_ids"] += len(gold_ids)
            if not missing and len(gold_ids) == len(article_ids):
                stats["cross_complete"] += 1
        else:
            article_id = article_ids[0] if article_ids else ""
            chunks = chunks_by_article.get(article_id, [])
            if not chunks:
                missing_sources.add(article_id or "<missing article_id>")
            if hinted_gold_ids:
                gold_ids = hinted_gold_ids
                stats["hinted_records"] += 1
            else:
                gold_ids = select_single_gold_ids(chunks, answers, top_k_single)
            record["gold_id"] = gold_ids
            stats["single_answerable"] += 1
            stats["single_gold_ids"] += len(gold_ids)
            if gold_ids:
                stats["single_mapped"] += 1

    stats["missing_sources"] = len(missing_sources)
    return {"counts": dict(stats), "missing_source_ids": sorted(missing_sources)}


def write_jsonl_atomic(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA_PATH)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNK_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL. Omit to update --qa atomically in place.",
    )
    parser.add_argument("--top-k-single", type=int, default=DEFAULT_TOP_K_SINGLE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k_single < 1:
        raise ValueError("--top-k-single must be at least 1")

    records = load_qa(args.qa)
    relevant_ids = {
        article_id
        for record in records
        for article_id in source_article_ids(record)
    }
    chunks_by_article, malformed_chunks = load_relevant_chunks(
        args.chunks, relevant_ids
    )
    result = add_gold_ids(records, chunks_by_article, args.top_k_single)
    counts = result["counts"]

    if result["missing_source_ids"]:
        preview = ", ".join(result["missing_source_ids"][:10])
        raise RuntimeError(
            f"Missing chunks for {counts['missing_sources']} source article(s): {preview}. "
            "The QA file was not modified."
        )

    print(f"QA records: {len(records)}")
    print(f"Relevant articles with chunks: {len(chunks_by_article)}/{len(relevant_ids)}")
    print(f"Malformed chunk lines skipped: {malformed_chunks}")
    print(
        "Single answerable: "
        f"{counts.get('single_mapped', 0)}/{counts.get('single_answerable', 0)} mapped, "
        f"{counts.get('single_gold_ids', 0)} gold IDs"
    )
    print(
        "Cross answerable: "
        f"{counts.get('cross_complete', 0)}/{counts.get('cross_answerable', 0)} cover all sources, "
        f"{counts.get('cross_gold_ids', 0)} gold IDs"
    )
    print(f"No-answer records: {counts.get('no_answer', 0)} -> gold_id=[]")

    if args.dry_run:
        print("Dry run: no file was modified.")
        return

    output_path = args.output or args.qa
    write_jsonl_atomic(records, output_path)
    # ascii() keeps Windows consoles with legacy encodings from failing on a
    # Vietnamese path after the atomic write has already succeeded.
    print(f"Saved: {ascii(str(output_path))}")


if __name__ == "__main__":
    main()
