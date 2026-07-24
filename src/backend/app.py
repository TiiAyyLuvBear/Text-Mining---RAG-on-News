import json
import logging
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import requests

ROOT = Path(__file__).resolve().parents[2]
RERANK_PATH = ROOT / "src" / "reranker" / "output" / "bge_token_output" / "rerank_token_bge_top5.jsonl"
QA_PATH = ROOT / "Dataset" / "QA_Claude" / "QA_output.jsonl"

load_dotenv(ROOT / ".env")

DEFAULT_HOST = os.getenv("RAG_API_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("RAG_API_PORT", "8000"))
ANTHROPIC_TIMEOUT = float(os.getenv("ANTHROPIC_TIMEOUT", "45"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2200"))
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "2.0"))
RERANK_MIN_MARGIN = float(os.getenv("RERANK_MIN_MARGIN", "2.0"))
LLM_API_URL = os.getenv("LLM_API_URL", "https://api.xah.io/v1/chat/completions")

logging.basicConfig(
    level=os.getenv("RAG_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("rag-backend")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize(text: str) -> str:
    return " ".join((text or "").lower().strip().split())

def answer_tokens(text: str) -> List[str]:
    return re.findall(r"\w+", normalize(text), flags=re.UNICODE)

def token_f1(prediction: str, reference: str) -> Optional[float]:
    pred_tokens = answer_tokens(prediction)
    ref_tokens = answer_tokens(reference)
    if not pred_tokens or not ref_tokens:
        return None

    ref_counts: Dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token in pred_tokens:
        if ref_counts.get(token, 0) > 0:
            overlap += 1
            ref_counts[token] -= 1

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

def load_gold_answers() -> Dict[str, str]:
    rows = load_jsonl(QA_PATH)
    gold_answers: Dict[str, str] = {}
    for row in rows:
        qa_id = str(row.get("id", ""))
        answers = row.get("answers") or row.get("plausible_answers") or []
        if qa_id and answers:
            gold_answers[qa_id] = str(answers[0])
    return gold_answers


def find_contexts(question: str, top_k: int) -> Tuple[str, List[Dict[str, Any]], float]:
    started = time.perf_counter()
    query_words = set(normalize(question).split())
    best_score = -1.0
    best_row: Optional[Dict[str, Any]] = None

    for row in load_jsonl(RERANK_PATH):
        row_question = normalize(str(row.get("question", "")))
        if not row_question:
            continue
        row_words = set(row_question.split())
        score = 1.0 if row_words == query_words else len(query_words & row_words) / max(len(query_words), 1)
        if score > best_score:
            best_score = score
            best_row = row

    if not best_row:
        LOGGER.warning("retrieval done | candidates=0 | elapsed_ms=%.1f", (time.perf_counter() - started) * 1000)
        return "runtime", [], 0.0

    candidates = best_row.get("reranked_candidates") or best_row.get("candidates") or []
    selected = candidates[:top_k]
    LOGGER.info(
        "retrieval done | qa_id=%s | candidates=%d/%d | confidence=%.3f | elapsed_ms=%.1f",
        best_row.get("qa_id", "runtime"), len(selected), len(candidates), max(0.0, min(best_score, 1.0)),
        (time.perf_counter() - started) * 1000,
    )
    return str(best_row.get("qa_id", "runtime")), selected, max(0.0, min(best_score, 1.0))


def evidence_quality(contexts: List[Dict[str, Any]]) -> Tuple[bool, float, float]:
    ranked = sorted(
        contexts,
        key=lambda item: float(item.get("rerank_score", float("-inf"))),
        reverse=True,
    )
    if not ranked:
        return False, float("-inf"), float("-inf")
    top_score = float(ranked[0].get("rerank_score", float("-inf")))
    top_article = str(ranked[0].get("article_id", ""))
    competing_scores = [
        float(item.get("rerank_score", float("-inf")))
        for item in ranked[1:]
        if str(item.get("article_id", "")) != top_article
    ]
    margin = top_score - max(competing_scores) if competing_scores else float("inf")
    return top_score >= RERANK_MIN_SCORE and margin >= RERANK_MIN_MARGIN, top_score, margin


def generate_answer(question: str, contexts: List[Dict[str, Any]]) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY in .env")

    model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4.8")
    context_parts = []
    used_chars = 0
    for idx, context in enumerate(contexts, start=1):
        text = str(context.get("text", ""))
        remaining = MAX_CONTEXT_CHARS - used_chars
        if remaining <= 0:
            break
        text = text[:remaining]
        context_parts.append(f"[{idx}] {text}")
        used_chars += len(text)
    context_text = "\n\n".join(context_parts)
    LOGGER.info("generation start | model=%s | contexts=%d | prompt_context_chars=%d | max_tokens=%d", model, len(context_parts), len(context_text), LLM_MAX_TOKENS)
    prompt = (
        "Bạn là hệ thống hỏi đáp RAG cho tin tức tiếng Việt. "
        "Hãy trả lời đầy đủ và có chiều sâu, không trả lời cụt ngủn. "
        "Chỉ sử dụng thông tin có trong CONTEXT; không được bịa hoặc suy diễn vượt quá bằng chứng. "
        "Hãy tổng hợp các context liên quan, nêu rõ nguyên nhân, diễn biến, tác động hoặc khuyến nghị "
        "nếu những thông tin đó có trong context. Ưu tiên các chi tiết cụ thể. "
        "Trình bày khoảng 3-6 đoạn hoặc danh sách 5-10 ý tùy câu hỏi. "
        "Nếu context không đủ bằng chứng, phải nói rõ phần nào chưa có dữ liệu.\n\n"
        "CONTEXT:\n" + context_text + "\n\nQUESTION:\n" + question + "\n\n"
        "Trả lời bằng tiếng Việt."
    )

    response = requests.post(
        LLM_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": 0.2,
        },
        timeout=ANTHROPIC_TIMEOUT,
    )
    if not response.ok:
        detail = response.text[:500].replace("\n", " ")
        raise RuntimeError(f"LLM API {response.status_code}: {detail}")

    data = response.json()
    try:
        answer = str(data["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("LLM API returned an invalid chat-completions response") from exc

    if not answer:
        raise RuntimeError("LLM API returned an empty answer")
    LOGGER.info("generation done | answer_chars=%d | finish_reason=%s", len(answer), data.get("choices", [{}])[0].get("finish_reason", "unknown"))
    return answer


class RagHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send_json(200, {"ok": True})
        LOGGER.info("API OPTIONS /ask -> 200")

    def do_POST(self) -> None:
        started = time.perf_counter()
        if self.path.rstrip("/") != "/ask":
            self._send_json(404, {"error": "Not found"})
            LOGGER.info("API POST %s -> 404", self.path)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            question = str(payload.get("question", "")).strip()
            top_k = int(payload.get("top_k", 5))
            if not question:
                self._send_json(400, {"error": "Missing question"})
                LOGGER.info("API POST /ask -> 400 (missing question)")
                return

            qa_id, contexts, confidence = find_contexts(question, top_k)
            sufficient, top_score, margin = evidence_quality(contexts)
            if sufficient:
                answer = generate_answer(question, contexts)
                answer_status = "generated"
            else:
                answer = "Không đủ thông tin trong dữ liệu được cung cấp để trả lời câu hỏi này một cách đáng tin cậy."
                answer_status = "abstained"
                LOGGER.warning(
                    "abstention | qa_id=%s | top_score=%.4f | margin=%.4f | min_score=%.2f | min_margin=%.2f",
                    qa_id, top_score, margin, RERANK_MIN_SCORE, RERANK_MIN_MARGIN,
                )
            reference_answer = load_gold_answers().get(qa_id, "")
            answer_accuracy = token_f1(answer, reference_answer) if reference_answer else None
            elapsed_ms = (time.perf_counter() - started) * 1000
            self._send_json(200, {
                "qa_id": qa_id,
                "answer": answer,
                "contexts": contexts,
                "confidence": 1.0 if sufficient else 0.0,
                "confidence_percent": 100.0 if sufficient else 0.0,
                "confidence_method": "BGE gate: top_score >= ngưỡng và top_score - second_score >= margin",
                "reference_answer": reference_answer,
                "answer_accuracy": answer_accuracy,
                "answer_accuracy_method": "token_f1_vs_gold_answer",
                "response_time_ms": round(elapsed_ms, 1),
                "evidence_sufficient": sufficient,
                "rerank_top_score": top_score,
                "rerank_margin": margin,
                "rerank_min_score": RERANK_MIN_SCORE,
                "rerank_min_margin": RERANK_MIN_MARGIN,
                "answer_status": answer_status,
            })
            LOGGER.info("API POST /ask -> 200 (%.0f ms, qa_id=%s)", elapsed_ms, qa_id)
        except Exception as exc:
            self._send_json(500, {"error": str(exc), "type": type(exc).__name__})
            elapsed_ms = (time.perf_counter() - started) * 1000
            LOGGER.error("API POST /ask -> 500 (%.0f ms, %s)", elapsed_ms, str(exc))


def main() -> None:
    server = ThreadingHTTPServer((DEFAULT_HOST, DEFAULT_PORT), RagHandler)
    LOGGER.info("Backend listening on http://%s:%s", DEFAULT_HOST, DEFAULT_PORT)
    LOGGER.info(
        "config | rerank_path=%s | rerank_exists=%s | qa_path=%s | qa_exists=%s",
        RERANK_PATH, RERANK_PATH.exists(), QA_PATH, QA_PATH.exists(),
    )
    LOGGER.info(
        "config | llm_endpoint=%s | llm_model=%s | max_tokens=%d | max_context_chars=%d | timeout_s=%.1f | rerank_min_score=%.2f | rerank_min_margin=%.2f",
        LLM_API_URL, os.getenv("ANTHROPIC_MODEL", "claude-opus-4.8"),
        LLM_MAX_TOKENS, MAX_CONTEXT_CHARS, ANTHROPIC_TIMEOUT, RERANK_MIN_SCORE, RERANK_MIN_MARGIN,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

