import argparse
import json
import os
import time
from pathlib import Path

try:
    from anthropic import Anthropic
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: anthropic. Install with: pip install anthropic"
    ) from exc


DEFAULT_INPUT = "Dataset/QA_Claude/QA_output.jsonl"
DEFAULT_OUTPUT = "src/NOT_RAG/answers_token_top5_claude_not_rag.jsonl"
DEFAULT_MODEL = "claude-opus-4.8"
DEFAULT_BASE_URL = "https://api.xah.io"

PROMPT_TEMPLATE = (
    "B\u1ea1n l\u00e0 h\u1ec7 th\u1ed1ng h\u1ecfi \u0111\u00e1p tin t\u1ee9c ti\u1ebfng Vi\u1ec7t.\n\n"
    "Y\u00eau c\u1ea7u:\n"
    "- Tr\u1ea3 l\u1eddi c\u00e2u h\u1ecfi ng\u1eafn g\u1ecdn, r\u00f5 r\u00e0ng, \u0111\u00fang tr\u1ecdng t\u00e2m.\n"
    "- N\u1ebfu kh\u00f4ng bi\u1ebft ho\u1eb7c kh\u00f4ng ch\u1eafc ch\u1eafn, tr\u1ea3 l\u1eddi \u0111\u00fang c\u00e2u sau: \"Kh\u00f4ng \u0111\u1ee7 th\u00f4ng tin \u0111\u1ec3 tr\u1ea3 l\u1eddi.\"\n\n"
    "C\u00e2u h\u1ecfi:\n{question}\n\n"
    "C\u00e2u tr\u1ea3 l\u1eddi:\n"
)

def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def load_done_ids(path):
    done = set()
    path = Path(path)
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("qa_id") is not None:
                done.add(row.get("qa_id"))
    return done


def extract_text(message):
    texts = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            texts.append(block.text)
    return "\n".join(texts).strip()


def call_llm(client, model, question, max_tokens, retries, retry_sleep):
    prompt = PROMPT_TEMPLATE.format(question=question)
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0,
                top_p=1,
                messages=[{"role": "user", "content": prompt}],
            )
            return extract_text(message)
        except Exception as exc:
            last_error = exc
            print("Call failed attempt " + str(attempt) + "/" + str(retries) + ": " + str(exc))
            time.sleep(retry_sleep)
    raise last_error


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate closed-book LLM-only answers without retrieval/RAG.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--resume", action="store_true", help="Skip qa_id already in output file.")
    return parser.parse_args()


def main():
    args = parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing ANTHROPIC_API_KEY environment variable. "
            "Set it first, for example in PowerShell: "
            "$env:ANTHROPIC_API_KEY='your_ckey_api_key'"
        )

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit("Input file not found: " + str(input_path))

    client = Anthropic(api_key=api_key, base_url=args.base_url)

    done_ids = load_done_ids(args.output) if args.resume else set()
    if args.resume and done_ids:
        print("Resume mode: skipping " + str(len(done_ids)) + " already done")

    processed = 0
    for index, row in enumerate(read_jsonl(input_path), start=1):
        if args.limit is not None and index > args.limit:
            break

        qa_id = row.get("qa_id") or row.get("id")
        if args.resume and qa_id in done_ids:
            continue

        generated_answer = call_llm(
            client=client,
            model=args.model,
            question=row.get("question", ""),
            max_tokens=args.max_tokens,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )

        output_row = {
            "qa_id": qa_id,
            "qa_type": row.get("qa_type"),
            "question": row.get("question"),
            "retrieval": "none",
            "reranker": "none",
            "llm_model": args.model,
            "generated_answer": generated_answer,
            "gold_articles": row.get("article_ids", []),
            "used_contexts": [],
        }

        append_jsonl(args.output, output_row)
        processed += 1

        if processed % 5 == 0:
            print("Generated " + str(processed) + " answers")

        time.sleep(args.sleep)

    print("Done. Wrote " + str(processed) + " new rows to " + str(args.output))


if __name__ == "__main__":
    main()


