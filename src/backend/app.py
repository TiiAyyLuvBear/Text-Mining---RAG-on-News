import json
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from anthropic import Anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
RERANK_PATH = ROOT / "src" / "re-ranker" / "output_Rerank" / "rerank_token_bge_top5.jsonl"

load_dotenv(ROOT / ".env")

DEFAULT_HOST = os.getenv("RAG_API_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("RAG_API_PORT", "8000"))
ANTHROPIC_TIMEOUT = float(os.getenv("ANTHROPIC_TIMEOUT", "45"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))


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


def find_contexts(question: str, top_k: int) -> Tuple[str, List[Dict[str, Any]], float]:
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
        return "runtime", [], 0.0

    candidates = best_row.get("reranked_candidates") or best_row.get("candidates") or []
    return str(best_row.get("qa_id", "runtime")), candidates[:top_k], max(0.0, min(best_score, 1.0))


def generate_answer(question: str, contexts: List[Dict[str, Any]]) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY in .env")

    base_url = os.getenv("ANTHROPIC_BASE_URL") or None
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
    prompt = (
        "Ban la he thong hoi dap RAG cho tin tuc tieng Viet. "
        "Chi tra loi dua tren context; neu khong du thong tin thi noi khong du du lieu.\n\n"
        "Context:\n" + context_text + "\n\nCau hoi: " + question
    )

    client = Anthropic(api_key=api_key, base_url=base_url, timeout=ANTHROPIC_TIMEOUT)
    message = client.messages.create(
        model=model,
        max_tokens=350,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in message.content if getattr(block, "type", "") == "text")


class RagHandler(BaseHTTPRequestHandler):
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

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/ask":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            question = str(payload.get("question", "")).strip()
            top_k = int(payload.get("top_k", 5))
            if not question:
                self._send_json(400, {"error": "Missing question"})
                return

            qa_id, contexts, confidence = find_contexts(question, top_k)
            answer = generate_answer(question, contexts)
            self._send_json(200, {"qa_id": qa_id, "answer": answer, "contexts": contexts, "confidence": confidence})
        except Exception as exc:
            traceback.print_exc()
            self._send_json(500, {"error": str(exc), "type": type(exc).__name__})


def main() -> None:
    server = ThreadingHTTPServer((DEFAULT_HOST, DEFAULT_PORT), RagHandler)
    print("RAG backend running at http://" + DEFAULT_HOST + ":" + str(DEFAULT_PORT) + "/ask")
    server.serve_forever()


if __name__ == "__main__":
    main()

