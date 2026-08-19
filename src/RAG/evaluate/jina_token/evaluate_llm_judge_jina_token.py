import argparse
import json
import os
import re
import time
from pathlib import Path
from statistics import mean
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

try:
    from anthropic import Anthropic
except ImportError as exc:
    raise SystemExit("Missing dependency: anthropic. Install with: pip install anthropic") from exc

DEFAULT_PRED = "data/generation/output/answers_token_jina_top5_claude.jsonl"
DEFAULT_GOLD = "Dataset/QA_Claude/QA_output.jsonl"
DEFAULT_OUT = "data/evaluate/output/jina_token/llm_judge_jina_token_gptscore.jsonl"
DEFAULT_SUMMARY = "data/evaluate/output/jina_token/llm_judge_jina_token_summary.json"
DEFAULT_MODEL = "claude-opus-4.8"
DEFAULT_BASE_URL = "https://api.xah.io"

JUDGE_PROMPT = """Bạn là giám khảo đánh giá chất lượng câu trả lời cho hệ thống hỏi đáp RAG tiếng Việt.

Đầu vào gồm Câu hỏi, Ground Truth Answer, Top-5 Context và Generated Answer.
Hãy chấm điểm Generated Answer theo 5 tiêu chí, mỗi tiêu chí từ 1 đến 5:
- Correctness: câu trả lời có đúng so với Ground Truth Answer hay không.
- Faithfulness: câu trả lời có bám sát Top-5 Context và không bịa ngoài context hay không.
- Completeness: câu trả lời có bao phủ đủ các ý quan trọng hay không.
- Relevance: câu trả lời có trả lời trực tiếp, đúng trọng tâm Câu hỏi hay không.
- Fluency: câu trả lời có tự nhiên, rõ ràng, dễ hiểu bằng tiếng Việt hay không.

Thang điểm:
1 = rất kém, 2 = kém, 3 = trung bình, 4 = tốt, 5 = rất tốt.
GPTScore là trung bình cộng của 5 tiêu chí.

Chỉ trả về JSON hợp lệ, không thêm bất kỳ nội dung nào ngoài JSON:
{{
  "correctness": 1-5,
  "faithfulness": 1-5,
  "completeness": 1-5,
  "relevance": 1-5,
  "fluency": 1-5,
  "gptscore": number,
  "reason": "lý do ngắn gọn"
}}

Câu hỏi:
{question}

Ground Truth Answer:
{reference_answer}

Top-5 Context:
{contexts}

Generated Answer:
{generated_answer}
"""

def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def load_gold(path):
    rows = {}
    for row in read_jsonl(path):
        qa_id = str(row.get("id"))
        answers = row.get("answers") or row.get("plausible_answers") or []
        rows[qa_id] = {
            "question": row.get("question", ""),
            "reference_answer": answers[0] if answers else "",
            "qa_type": row.get("qa_type"),
        }
    return rows


def build_context(contexts):
    parts = []
    for item in contexts[:5]:
        parts.append(
            "[Context {rank}]\nArticle ID: {article_id}\nChunk ID: {chunk_id}\nRerank score: {score}\nText:\n{text}".format(
                rank=item.get("rank"),
                article_id=item.get("article_id"),
                chunk_id=item.get("chunk_id"),
                score=item.get("rerank_score"),
                text=item.get("text", ""),
            )
        )
    return "\n\n".join(parts)


def load_done(path):
    done = set()
    path = Path(path)
    if not path.exists():
        return done
    for row in read_jsonl(path):
        if row.get("qa_id") is not None:
            done.add(str(row.get("qa_id")))
    return done


def extract_text(message):
    content = getattr(message, "content", None)
    if content is None:
        stop_reason = getattr(message, "stop_reason", None)
        request_id = getattr(message, "_request_id", None)
        details = []
        if stop_reason is not None:
            details.append("stop_reason=" + str(stop_reason))
        if request_id is not None:
            details.append("request_id=" + str(request_id))
        suffix = " (" + ", ".join(details) + ")" if details else ""
        raise ValueError("LLM response has no content" + suffix)

    texts = []
    for block in content:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", None)
            if text:
                texts.append(text)

    answer = "\n".join(texts).strip()
    if not answer:
        raise ValueError("LLM response contains no text blocks")
    return answer


def parse_json(text):
    if not text or not text.strip():
        raise ValueError("Judge returned empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def normalize_score(value):
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(1.0, min(5.0, score))


def judge_one(client, model, prompt, max_tokens, retries, retry_sleep):
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
            return parse_json(extract_text(message))
        except Exception as exc:
            last_error = exc
            print("Judge failed attempt " + str(attempt) + "/" + str(retries) + ": " + str(exc))
            if attempt < retries:
                time.sleep(retry_sleep)
    raise RuntimeError("Judge failed after retries: " + str(last_error)) from last_error


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(out_path, rows):
    criteria = ["correctness", "faithfulness", "completeness", "relevance", "fluency", "gptscore"]
    summary = {"num_judged": len(rows)}
    for key in criteria:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        summary["avg_" + key] = mean(values) if values else 0.0
    Path(out_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated answers with LLM-as-a-Judge GPTScore.")
    parser.add_argument("--pred", default=DEFAULT_PRED)
    parser.add_argument("--gold", default=DEFAULT_GOLD)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Missing ANTHROPIC_API_KEY environment variable")

    gold_by_id = load_gold(args.gold)
    done = load_done(args.output) if args.resume else set()
    if args.resume:
        print("Resume mode: skipping " + str(len(done)) + " already judged qa_id values")
    client = Anthropic(api_key=api_key, base_url=args.base_url, timeout=120.0)

    processed = 0
    for pred in read_jsonl(args.pred):
        if args.limit is not None and processed >= args.limit:
            break
        qa_id = str(pred.get("qa_id"))
        if qa_id in done:
            continue
        gold = gold_by_id.get(qa_id)
        if not gold:
            continue

        prompt = JUDGE_PROMPT.format(
            question=pred.get("question") or gold.get("question", ""),
            reference_answer=gold.get("reference_answer", ""),
            contexts=build_context(pred.get("used_contexts", [])),
            generated_answer=pred.get("generated_answer", ""),
        )
        try:
            result = judge_one(client, args.model, prompt, args.max_tokens, args.retries, args.retry_sleep)
        except Exception as exc:
            print("Skipping qa_id " + qa_id + " after retries: " + str(exc))
            continue

        scores = {
            key: normalize_score(result.get(key))
            for key in ["correctness", "faithfulness", "completeness", "relevance", "fluency"]
        }
        gptscore = sum(scores.values()) / 5
        row = {
            "qa_id": qa_id,
            "qa_type": pred.get("qa_type") or gold.get("qa_type"),
            "judge_model": args.model,
            **scores,
            "gptscore": gptscore,
            "reason": result.get("reason", ""),
        }
        append_jsonl(args.output, row)
        done.add(qa_id)
        processed += 1
        print("Judged " + str(processed) + ": " + qa_id + " GPTScore=" + str(round(gptscore, 3)))
        time.sleep(args.sleep)

    rows = list(read_jsonl(args.output)) if Path(args.output).exists() else []
    summary = write_summary(args.summary, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



