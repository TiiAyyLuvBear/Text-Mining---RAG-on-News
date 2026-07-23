import json
import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

EMBEDDING_PATH = ROOT / "src" / "embedding" / "output" / "per_query_token.jsonl"
RERANK_PATH = ROOT / "src" / "re-ranker" / "output_Rerank" / "rerank_token_bge_top5.jsonl"
LLM_PATH = ROOT / "src" / "LLM_OUTPUT" / "BGE_TOKEN" / "answers_token_bge_top5_claude.jsonl"
QA_PATH = ROOT / "Dataset" / "QA_Claude" / "QA_output.jsonl"

DEFAULT_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000/api/qa/ask")

st.set_page_config(
    page_title="RAG News QA | BGE Token",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top left, #eff6ff 0, #f8fafc 34%, #ffffff 100%);
        }
        .hero {
            padding: 1.5rem 1.7rem;
            border-radius: 28px;
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 54%, #06b6d4 100%);
            color: white;
            box-shadow: 0 22px 60px rgba(30, 64, 175, 0.22);
            margin-bottom: 1.2rem;
        }
        .hero h1 {
            margin: 0 0 .45rem 0;
            font-size: 2.45rem;
            line-height: 1.1;
        }
        .hero p {
            margin: 0;
            color: #dbeafe;
            font-size: 1.02rem;
        }
        .glass-card {
            padding: 1rem 1.1rem;
            border: 1px solid rgba(148, 163, 184, .28);
            border-radius: 22px;
            background: rgba(255, 255, 255, .82);
            box-shadow: 0 14px 35px rgba(15, 23, 42, .08);
        }
        .answer-card {
            padding: 1.2rem 1.35rem;
            border-radius: 24px;
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border-left: 6px solid #2563eb;
            box-shadow: 0 18px 45px rgba(37, 99, 235, .11);
            font-size: 1.08rem;
            line-height: 1.68;
        }
        .flow-step {
            padding: .8rem .9rem;
            border-radius: 18px;
            background: white;
            border: 1px solid #e2e8f0;
            min-height: 96px;
        }
        .flow-title {
            color: #0f172a;
            font-weight: 750;
            font-size: .98rem;
            margin-bottom: .25rem;
        }
        .flow-subtitle {
            color: #64748b;
            font-size: .84rem;
        }
        .badge {
            display: inline-block;
            padding: .18rem .55rem;
            border-radius: 999px;
            background: #dbeafe;
            color: #1d4ed8;
            font-size: .78rem;
            font-weight: 700;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@st.cache_data(show_spinner=False)
def load_data() -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, str]]:
    embedding_rows = load_jsonl(EMBEDDING_PATH)
    rerank_rows = load_jsonl(RERANK_PATH)
    llm_rows = load_jsonl(LLM_PATH)
    qa_rows = load_jsonl(QA_PATH)
    rerank_by_id = {str(row.get("qa_id")): row for row in rerank_rows}
    llm_by_id = {str(row.get("qa_id")): row for row in llm_rows}
    gold_by_id = {}
    for row in qa_rows:
        answers = row.get("answers") or row.get("plausible_answers") or []
        if answers:
            gold_by_id[str(row.get("id"))] = str(answers[0])
    return embedding_rows, rerank_by_id, llm_by_id, gold_by_id


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


def find_best_question(question: str, rows: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], float]:
    query = normalize(question)
    if not query:
        return None, 0.0
    exact = [row for row in rows if normalize(str(row.get("question", ""))) == query]
    if exact:
        return exact[0], 1.0
    scored = []
    for row in rows:
        source = normalize(str(row.get("question", "")))
        score = SequenceMatcher(None, query, source).ratio()
        if query in source or source in query:
            score = max(score, 0.92)
        scored.append((score, row))
    if not scored:
        return None, 0.0
    score, row = max(scored, key=lambda item: item[0])
    return row, score


def build_fallback_answer(rerank_row: Optional[Dict[str, Any]]) -> str:
    if not rerank_row:
        return "Không tìm thấy dữ liệu phù hợp trong luồng BGE token đã có."
    candidates = rerank_row.get("reranked_candidates") or rerank_row.get("candidates") or []
    if not candidates:
        return "Không đủ thông tin trong dữ liệu được cung cấp."
    top_text = str(candidates[0].get("text", "")).strip()
    if not top_text:
        return "Không đủ thông tin trong dữ liệu được cung cấp."
    return top_text[:900] + ("..." if len(top_text) > 900 else "")


def resolve_answer(llm_row: Optional[Dict[str, Any]], rerank_row: Optional[Dict[str, Any]]) -> str:
    if llm_row:
        for key in ("answer", "llm_answer", "response", "prediction", "generated_answer"):
            value = llm_row.get(key)
            if value:
                return str(value)
    return build_fallback_answer(rerank_row)


def get_candidates(row: Optional[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if not row:
        return []
    candidates = row.get("reranked_candidates") or row.get("candidates") or []
    return candidates[:limit]


def call_backend(api_url: str, question: str, top_k: int) -> Optional[Dict[str, Any]]:
    if not requests:
        st.warning("Chưa cài `requests`, UI sẽ dùng dữ liệu offline.")
        return None
    try:
        response = requests.post(api_url, json={"question": question, "top_k": top_k}, timeout=70)
        if not response.ok:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            st.warning(f"Backend loi {response.status_code}: {detail}")
            return None
        return response.json()
    except Exception as exc:
        st.warning(f"Không gọi được backend, chuyển sang offline cache: {exc}")
        return None


def render_flow(embedding_row: Optional[Dict[str, Any]], rerank_row: Optional[Dict[str, Any]], llm_row: Optional[Dict[str, Any]]) -> None:
    embed_candidates = len(embedding_row.get("candidates", [])) if embedding_row else 0
    rerank_candidates = len(get_candidates(rerank_row, 999))
    llm_status = "Có output" if llm_row else "Fallback context"
    cols = st.columns(3)
    steps = [
        ("1", "Embedding token", f"Truy xuất {embed_candidates} chunks từ `per_query_token.jsonl`"),
        ("2", "Rerank BGE token", f"Sắp hạng lại top {rerank_candidates} chunks bằng BGE reranker"),
        ("3", "LLM output", f"Sinh câu trả lời từ top context · {llm_status}"),
    ]
    for col, (idx, title, subtitle) in zip(cols, steps):
        col.markdown(
            f"""
            <div class="flow-step">
                <span class="badge">Step {idx}</span>
                <div class="flow-title">{title}</div>
                <div class="flow-subtitle">{subtitle}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


embedding_rows, rerank_by_id, llm_by_id, gold_by_id = load_data()

st.markdown(
    """
    <div class="hero">
        <h1>📰 Vietnamese News RAG QA</h1>
        <p>Giao diện hỏi đáp theo luồng <b>Embedding Token → BGE Rerank Token → LLM Output</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Cấu hình")
    mode = st.radio("Nguồn trả lời", ["Offline cache", "Backend API"], index=0)
    top_k = st.slider("Số context hiển thị", min_value=1, max_value=10, value=5)
    similarity_threshold = st.slider("Ngưỡng match câu hỏi offline", 0.1, 1.0, 0.45, 0.05)
    api_url = DEFAULT_API_URL
    st.divider()
    if st.button("🔄 Reload dữ liệu"):
        st.cache_data.clear()
        st.rerun()

sample_questions = [str(row.get("question", "")) for row in embedding_rows[:80] if row.get("question")]
selected_sample = st.selectbox("Chọn nhanh câu hỏi mẫu", [""] + sample_questions)
question = st.text_area(
    "Nhập câu hỏi của người dùng",
    value=selected_sample,
    height=115,
    placeholder="Ví dụ: Nước hầm xương nấu phở có tiềm ẩn nguy cơ gây hại không?",
)

ask = st.button("🚀 Hỏi hệ thống RAG", type="primary", use_container_width=True)

if ask:
    if not question.strip():
        st.error("Bạn hãy nhập câu hỏi trước nha.")
        st.stop()

    started = time.perf_counter()
    backend_payload = None
    if mode == "Backend API":
        with st.spinner("Đang gọi backend RAG..."):
            backend_payload = call_backend(api_url, question, top_k)

    if backend_payload:
        answer = str(backend_payload.get("answer") or backend_payload.get("response") or "")
        candidates = backend_payload.get("contexts") or backend_payload.get("candidates") or []
        qa_id = str(backend_payload.get("qa_id", "runtime"))
        match_score = float(backend_payload.get("confidence", 1.0))
        reference_answer = str(backend_payload.get("reference_answer") or gold_by_id.get(qa_id, ""))
        answer_accuracy = backend_payload.get("answer_accuracy")
        if answer_accuracy is None and reference_answer:
            answer_accuracy = token_f1(answer, reference_answer)
        embedding_row = {"qa_id": qa_id, "question": question, "candidates": candidates}
        rerank_row = {"qa_id": qa_id, "reranked_candidates": candidates}
        llm_row = {"qa_id": qa_id, "answer": answer}
    else:
        embedding_row, match_score = find_best_question(question, embedding_rows)
        if not embedding_row or match_score < similarity_threshold:
            st.error("Không tìm thấy câu hỏi đủ gần trong dữ liệu offline. Hãy chọn câu hỏi mẫu hoặc chạy backend API.")
            st.stop()
        qa_id = str(embedding_row.get("qa_id"))
        rerank_row = rerank_by_id.get(qa_id)
        llm_row = llm_by_id.get(qa_id)
        answer = resolve_answer(llm_row, rerank_row)
        reference_answer = gold_by_id.get(qa_id, "")
        answer_accuracy = token_f1(answer, reference_answer) if reference_answer else None
        candidates = get_candidates(rerank_row, top_k)

    elapsed = time.perf_counter() - started
    st.session_state["last_result"] = {
        "qa_id": qa_id,
        "question": question,
        "answer": answer,
        "candidates": candidates,
        "embedding_row": embedding_row,
        "rerank_row": rerank_row,
        "llm_row": llm_row,
        "match_score": match_score,
        "reference_answer": reference_answer,
        "answer_accuracy": answer_accuracy,
        "elapsed": elapsed,
    }

result = st.session_state.get("last_result")
if result:
    qa_id = result["qa_id"]
    embedding_row = result["embedding_row"]
    rerank_row = result["rerank_row"]
    llm_row = result["llm_row"]
    candidates = result["candidates"]

    st.divider()

    answer_accuracy = result.get("answer_accuracy")
    accuracy_display = "N/A"
    if answer_accuracy is not None:
        accuracy_percent = round(max(0.0, min(float(answer_accuracy), 1.0)) * 100, 1)
        accuracy_display = str(accuracy_percent) + "%"
    metric_cols = st.columns(4)
    metric_cols[0].metric("QA ID", qa_id)
    metric_cols[1].metric("Độ chính xác", accuracy_display)
    metric_cols[2].metric("Top context", len(candidates))
    metric_cols[3].metric("Thời gian", str(round(result["elapsed"], 2)) + "s")

    st.subheader("💬 Câu trả lời")
    st.markdown(f"<div class='answer-card'>{result['answer']}</div>", unsafe_allow_html=True)

    tab_context, tab_embedding, tab_rerank, tab_raw = st.tabs(
        ["📚 Context sau rerank", "🔎 Embedding token", "🏆 BGE rerank token", "🧾 Raw JSON"]
    )

    with tab_context:
        if not candidates:
            st.info("Không có context rerank để hiển thị.")
        for idx, candidate in enumerate(candidates, start=1):
            title = f"#{idx} · Article {candidate.get('article_id', 'N/A')} · Chunk {candidate.get('chunk_id', 'N/A')}"
            score = candidate.get("rerank_score", candidate.get("score", candidate.get("retrieval_score", "N/A")))
            with st.expander(title + " - score: " + str(score), expanded=idx == 1):
                st.write(candidate.get("text", ""))
                st.json({k: v for k, v in candidate.items() if k != "text"}, expanded=False)

    with tab_embedding:
        st.markdown("Dữ liệu truy xuất ban đầu từ `src/embedding/output/per_query_token.jsonl`.")
        embed_candidates = get_candidates(embedding_row, top_k)
        for idx, candidate in enumerate(embed_candidates, start=1):
            embed_score = candidate.get("score", candidate.get("retrieval_score", "N/A"))
            with st.expander(f"Embedding candidate #{idx} - retrieval_score: {embed_score}"):
                st.write(candidate.get("text", ""))
                st.json({k: v for k, v in candidate.items() if k != "text"}, expanded=False)

    with tab_rerank:
        st.markdown("Dữ liệu top-k sau rerank từ `src/re-ranker/output_Rerank/rerank_token_bge_top5.jsonl`.")
        if rerank_row and rerank_row.get("rerank_metrics"):
            st.json(rerank_row.get("rerank_metrics"), expanded=True)
        for idx, candidate in enumerate(candidates, start=1):
            st.markdown(f"**Rank {candidate.get('rank', idx)}** · rerank_score `{candidate.get('rerank_score', 'N/A')}`")
            st.caption(str(candidate.get("text", ""))[:500])

    with tab_raw:
        raw_cols = st.columns(3)
        raw_cols[0].json(embedding_row, expanded=False)
        raw_cols[1].json(rerank_row or {}, expanded=False)
        raw_cols[2].json(llm_row or {}, expanded=False)
else:
    st.info("Nhập câu hỏi rồi bấm **Hỏi hệ thống RAG** để xem luồng xử lý.")
