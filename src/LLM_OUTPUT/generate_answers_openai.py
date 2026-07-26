"""Generate BGE-structure RAG answers with an OpenAI-compatible API."""
import argparse
import json
import os
import time
from pathlib import Path

try:
    import requests
except ImportError as exc:
    raise SystemExit("Missing dependency: requests. Install with: pip install requests") from exc


DEFAULT_INPUT = "src/re-ranker/output_Rerank/rerank_structure_bge_top5.jsonl"
DEFAULT_OUTPUT = "src/LLM_OUTPUT/answers_structure_bge_top5_openai.jsonl"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


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


def build_context(candidates):
    parts = []
    for candidate in candidates:
        block = (
            "[Context " + str(candidate.get("rank")) + "]\n"
            "Article ID: " + str(candidate.get("article_id")) + "\n"
            "Chunk ID: " + str(candidate.get("chunk_id")) + "\n"
            "Rerank score: " + str(candidate.get("rerank_score")) + "\n"
            "Text:\n" + str(candidate.get("text", "")) + "\n"
        )
        parts.append(block)
    return "\n".join(parts)


def chat_completions_url(base_url):
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return url + "/chat/completions"


def extract_text(data):
    content = data["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in (None, "text"):
                texts.append(str(block.get("text", "")))
        return "\n".join(texts).strip()
    raise RuntimeError("Unsupported chat-completions content format")


def call_llm(api_key, base_url, model, question, contexts, max_tokens, retries, retry_sleep, timeout):
    prompt = PROMPT_TEMPLATE.format(question=question, contexts=contexts)
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                chat_completions_url(base_url),
                headers={
                    "Authorization": "Bearer " + api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "top_p": 1,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=timeout,
            )
            if not response.ok:
                raise RuntimeError(
                    "LLM API " + str(response.status_code) + ": " + response.text[:500]
                )
            answer = extract_text(response.json())
            if not answer:
                raise RuntimeError("LLM API returned an empty answer")
            return answer
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
    parser = argparse.ArgumentParser(
        description="Generate structured BGE rerank answers with an OpenAI-compatible model."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--top-n-context", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true", help="Skip qa_id already in output file.")
    return parser.parse_args()


def main():
    args = parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit("Missing " + args.api_key_env + " environment variable.")

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit("Input file not found: " + str(input_path))

    done_ids = load_done_ids(args.output) if args.resume else set()
    if args.resume and done_ids:
        print("Resume mode: skipping " + str(len(done_ids)) + " already done")

    processed = 0
    for index, row in enumerate(read_jsonl(input_path), start=1):
        if args.limit is not None and index > args.limit:
            break

        qa_id = row.get("qa_id")
        if args.resume and qa_id in done_ids:
            continue

        top_contexts = row.get("reranked_candidates", [])[: args.top_n_context]
        contexts_text = build_context(top_contexts)

        generated_answer = call_llm(
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            question=row.get("question", ""),
            contexts=contexts_text,
            max_tokens=args.max_tokens,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
            timeout=args.timeout,
        )

        output_row = {
            "qa_id": qa_id,
            "qa_type": row.get("qa_type"),
            "question": row.get("question"),
            "embedding_type": row.get("embedding_type"),
            "reranker": row.get("reranker"),
            "llm_model": args.model,
            "top_n_context": args.top_n_context,
            "generated_answer": generated_answer,
            "gold_articles": row.get("gold_articles", []),
            "used_contexts": top_contexts,
            "rerank_metrics": row.get("rerank_metrics", {}),
        }

        append_jsonl(args.output, output_row)
        processed += 1

        if processed % 5 == 0:
            print("Generated " + str(processed) + " answers")

        time.sleep(args.sleep)

    print("Done. Wrote " + str(processed) + " new rows to " + str(args.output))


if __name__ == "__main__":
    main()