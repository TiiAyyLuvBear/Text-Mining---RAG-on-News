"""Repair generated-answer JSONL files without regenerating completed answers.

The repair workflow is intentionally conservative:
1. Keep the first row for each qa_id and remove later duplicates.
2. Compare the repaired output with its reranker input.
3. Call the LLM only for qa_ids missing from the output.

Examples (run from the repository root):
    python src/RAG/generation/repair_generation_outputs.py --dry-run
    python src/RAG/generation/repair_generation_outputs.py
    python src/RAG/generation/repair_generation_outputs.py --only token_bge
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

try:
    from anthropic import Anthropic
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit("Missing dependency: anthropic. Install with: pip install anthropic") from exc


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "src" / "generation" / "output"

TARGETS = {
    "structure_bge": {
        "output": OUTPUT_DIR / "answers_structure_bge_top5_claude.jsonl",
        "input": ROOT / "src" / "reranker" / "output" / "bge_structure_output" / "rerank_structured_bge_top5.jsonl",
    },
    "structure_jina": {
        "output": OUTPUT_DIR / "answers_structure_jina_top5_claude.jsonl",
        "input": ROOT / "src" / "reranker" / "output" / "jina_stucture_output" / "rerank_structure_jina_top5.jsonl",
    },
    "token_bge": {
        "output": OUTPUT_DIR / "answers_token_bge_top5_claude.jsonl",
        "input": ROOT / "src" / "reranker" / "output" / "bge_token_output" / "rerank_token_bge_top5.jsonl",
    },
    "token_jina": {
        "output": OUTPUT_DIR / "answers_token_jina_top5_claude.jsonl",
        "input": ROOT / "src" / "reranker" / "output" / "jina_token_output" / "rerank_token_jina_top5.jsonl",
    },
}

PROMPT_TEMPLATE = (
    "Bạn là hệ thống hỏi đáp dựa trên tin tức tiếng Việt.\n\n"
    "Yêu cầu:\n"
    "- Chỉ trả lời dựa trên Context được cung cấp.\n"
    "- Không tự suy đoán, không bổ sung kiến thức ngoài Context.\n"
    "- Nếu Context không đủ thông tin để trả lời, trả lời đúng câu sau: \"Không đủ thông tin trong dữ liệu được cung cấp.\"\n"
    "- Trả lời ngắn gọn, rõ ràng, đúng trọng tâm.\n\n"
    "Câu hỏi:\n{question}\n\n"
    "Context:\n{contexts}\n\n"
    "Câu trả lời:\n"
)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_number}")
            yield row


def load_rows(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path)) if path.exists() else []


def deduplicate_rows(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    duplicates: list[str] = []
    for row in rows:
        qa_id = row.get("qa_id")
        if qa_id is None:
            kept.append(row)
            continue
        qa_id = str(qa_id)
        if qa_id in seen:
            duplicates.append(qa_id)
            continue
        seen.add(qa_id)
        kept.append(row)
    return kept, duplicates


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        for row in rows:
            temporary.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def build_context(candidates: list[dict[str, Any]]) -> str:
    blocks = []
    for candidate in candidates:
        blocks.append(
            f"[Context {candidate.get('rank')}]\n"
            f"Article ID: {candidate.get('article_id')}\n"
            f"Chunk ID: {candidate.get('chunk_id')}\n"
            f"Rerank score: {candidate.get('rerank_score')}\n"
            f"Text:\n{candidate.get('text', '')}\n"
        )
    return "\n".join(blocks)


def extract_text(message: Any) -> str:
    content = getattr(message, "content", None)
    texts = []
    if content:
        if isinstance(content, str):
            return content.strip()
        for block in content:
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                texts.append(block.text)
    if texts:
        return "\n".join(texts).strip()
    raise RuntimeError(f"API returned no usable text: {message!r}")


def generate_answer(client: Any, args: argparse.Namespace, row: dict[str, Any]) -> str:
    prompt = PROMPT_TEMPLATE.format(
        question=row.get("question", ""),
        contexts=build_context(row.get("reranked_candidates", [])[: args.top_n_context]),
    )
    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            response = client.messages.create(
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=0,
                top_p=1,
                messages=[{"role": "user", "content": prompt}],
            )
            return extract_text(response)
        except Exception as exc:  # API/network errors should be retried
            last_error = exc
            print(f"  API attempt {attempt}/{args.retries} failed: {exc}")
            if attempt < args.retries:
                time.sleep(args.retry_sleep)
    assert last_error is not None
    raise last_error


def make_output_row(row: dict[str, Any], answer: str, model: str, top_n_context: int) -> dict[str, Any]:
    contexts = row.get("reranked_candidates", [])[:top_n_context]
    return {
        "qa_id": row.get("qa_id"),
        "qa_type": row.get("qa_type"),
        "question": row.get("question"),
        "embedding_type": row.get("embedding_type"),
        "reranker": row.get("reranker"),
        "llm_model": model,
        "top_n_context": top_n_context,
        "generated_answer": answer,
        "gold_articles": row.get("gold_articles", []),
        "used_contexts": contexts,
        "rerank_metrics": row.get("rerank_metrics", {}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=[*TARGETS, "all"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="Report repairs without rewriting or calling the API.")
    parser.add_argument("--model", default="claude-opus-4.8")
    parser.add_argument("--base-url", default="https://api.xah.io")
    parser.add_argument("--top-n-context", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    names = list(TARGETS) if args.only == "all" else [args.only]
    client = None

    for name in names:
        target = TARGETS[name]
        output_path = target["output"]
        input_path = target["input"]
        output_rows = load_rows(output_path)
        input_rows = load_rows(input_path)
        repaired_rows, duplicates = deduplicate_rows(output_rows)
        done_ids = {str(row["qa_id"]) for row in repaired_rows if row.get("qa_id") is not None}
        input_by_id: dict[str, dict[str, Any]] = {}
        for row in input_rows:
            if row.get("qa_id") is not None:
                input_by_id.setdefault(str(row["qa_id"]), row)
        missing_ids = [qa_id for qa_id in input_by_id if qa_id not in done_ids]

        print(f"[{name}] rows={len(output_rows)} unique={len(done_ids)} duplicates={len(duplicates)} missing={len(missing_ids)}")
        if duplicates:
            print(f"  duplicate qa_id: {', '.join(sorted(set(duplicates)))}")
        if missing_ids:
            print(f"  missing qa_id: {', '.join(missing_ids)}")

        if args.dry_run:
            continue

        if duplicates:
            write_jsonl_atomic(output_path, repaired_rows)
            print(f"  deduplicated: wrote {len(repaired_rows)} rows")

        if not missing_ids:
            continue
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("Missing ANTHROPIC_API_KEY; set it before filling missing answers.")
        if client is None:
            client = Anthropic(api_key=api_key, base_url=args.base_url)

        with output_path.open("a", encoding="utf-8") as output_file:
            for qa_id in missing_ids:
                row = input_by_id[qa_id]
                try:
                    answer = generate_answer(client, args, row)
                    output_file.write(json.dumps(make_output_row(row, answer, args.model, args.top_n_context), ensure_ascii=False) + "\n")
                    output_file.flush()
                    print(f"  generated qa_id={qa_id}")
                except Exception as exc:
                    print(f"  skipped qa_id={qa_id}: {exc}")
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()

