import pandas as pd
import openai
import json
import os
import sys
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSONL   = os.path.join(SCRIPT_DIR, "QA_output_new_480_update.jsonl")
CHUNKS_JSONL   = os.path.join(SCRIPT_DIR, "vieonline_news_chunks_token.jsonl")
EXISTING_QA_JSONL = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "QA_Claude", "QA_output.jsonl")
)

# API config
API_KEY   = "sk-586af3ccf3799c4de3187d05cbc4bdc7d64216e80cdac66291e932ec2db574d1"
BASE_URL  = "https://api.xah.io/v1"   # <-- SỬA LẠI BASE_URL NẾU CẦN
MODEL     = "claude-opus-4.6"

BATCH_SIZE  = 5
PHASE1_SKIP_ROWS = 20       # Bỏ qua 20 bài đầu đã lấy ở lần trước
PHASE1_ARTICLE_COUNT = 80   # Lấy 60 bài chưa có trong các file QA
QUESTIONS_PER_ARTICLE = 5
CROSS_ARTICLE_GROUP_SIZE = 3   # Số bài báo gom lại để tạo câu hỏi cross-article
CROSS_ARTICLE_QA_COUNT = 2    # Số câu hỏi cross-article mỗi nhóm
MIN_SIMILARITY = 0.15         # Ngưỡng similarity tối thiểu để nhóm bài lại
MAX_CROSS_GROUPS_PER_CATEGORY = 2  # Mỗi category chỉ lấy tối đa N nhóm tốt nhất
SAMPLE_PER_CATEGORY = 50           # Mỗi category chỉ sample N bài để clustering (tránh chậm)

MIN_SINGLE_CHUNKS_FOR_DIVERSITY = 5
MAX_SINGLE_CONTEXT_CHUNKS = 5
MAX_CROSS_CONTEXT_CHUNKS_PER_ARTICLE = 2

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
ARTICLE_CHUNKS: dict[str, list[dict]] = {}

SINGLE_DIVERSITY_PROMPT = """Yêu cầu phân bố nguồn bắt buộc:
- Nội dung bài báo được chia thành các đoạn có nhãn CHUNK_ID theo thứ tự từ đầu đến cuối bài.
- Hãy phân bố 5 câu QA trên nhiều chunk khác nhau; nếu có từ 3 chunk trở lên thì các câu có đáp án phải sử dụng thông tin từ ít nhất 3 chunk khác nhau.
- Không được mặc định chỉ lấy dữ kiện ở chunk đầu. Ưu tiên cả phần đầu, giữa và cuối bài.
- Không nhắc CHUNK_ID trong câu hỏi hoặc câu trả lời; CHUNK_ID chỉ là nhãn định vị nguồn.
"""

CROSS_DIVERSITY_PROMPT = """Yêu cầu phân bố nguồn bắt buộc:
- Mỗi bài được trình bày bằng các đoạn có nhãn CHUNK_ID rải từ đầu đến cuối bài.
- Mỗi câu cross-article có đáp án phải tổng hợp thông tin từ TẤT CẢ bài nguồn.
- Khi chọn bằng chứng trong từng bài, không mặc định chỉ dùng chunk đầu; ưu tiên các vị trí khác nhau của bài nếu nội dung cho phép.
- Không nhắc CHUNK_ID trong câu hỏi hoặc câu trả lời.
"""

# ── QA Taxonomy ─────────────────────────────────────────────────────────────
# Tỉ lệ mục tiêu cho RAG evaluation:
#   40% factoid / event_summary
#   30% cause_effect / entity_role / comparison
#   20% multi_doc_comparison / timeline  (Phase 2 - cross-article)
#   10% unanswerable / claim_verification
#
# Single-article (5 câu): 2 factoid/event_summary, 2 cause_effect/entity_role/comparison, 1 claim_verification/unanswerable
# Cross-article  (2 câu): multi_doc_comparison / timeline

QA_TAXONOMY = {
    "factoid": {
        "description": "Hỏi ai, cái gì, khi nào, ở đâu, số liệu cụ thể",
        "reasoning": "thấp",
        "example": "Sự kiện xảy ra vào ngày nào? / Ai là người phát biểu?"
    },
    "event_summary": {
        "description": "Tóm tắt sự kiện chính trong bài",
        "reasoning": "trung bình",
        "example": "Sự kiện chính được đề cập trong bài là gì?"
    },
    "cause_effect": {
        "description": "Hỏi nguyên nhân hoặc hệ quả của sự kiện",
        "reasoning": "trung bình",
        "example": "Nguyên nhân dẫn đến X là gì? / Hệ quả của Y là gì?"
    },
    "entity_role": {
        "description": "Vai trò của cá nhân/tổ chức trong bài",
        "reasoning": "trung bình",
        "example": "Vai trò của tổ chức A trong sự kiện này là gì?"
    },
    "comparison": {
        "description": "So sánh hai đối tượng trong cùng bài",
        "reasoning": "trung bình",
        "example": "Điểm khác biệt giữa phương án A và B là gì?"
    },
    "claim_verification": {
        "description": "Kiểm chứng một nhận định đúng/sai dựa trên bài",
        "reasoning": "cao",
        "example": "Đúng hay sai: 'X đã xảy ra trước Y'? Dẫn chứng?"
    },
    "unanswerable": {
        "description": "Câu hỏi mà bài báo KHÔNG đủ thông tin để trả lời",
        "reasoning": "cao",
        "example": "Cảm xúc cá nhân / dự đoán tương lai / thông tin chưa đề cập"
    },
    "multi_doc_comparison": {
        "description": "So sánh thông tin từ nhiều bài báo",
        "reasoning": "cao",
        "example": "Điểm chung giữa sự kiện ở bài 1 và bài 2?"
    },
    "timeline": {
        "description": "Diễn biến theo thời gian, cần tổng hợp nhiều nguồn",
        "reasoning": "cao",
        "example": "Trình tự thời gian các sự kiện diễn ra như thế nào?"
    },
}

SYSTEM_PROMPT = """Bạn là chuyên gia tạo bộ câu hỏi – câu trả lời (QA) từ bài báo tiếng Việt, phục vụ đánh giá hệ thống RAG.

Với mỗi bài báo, hãy sinh đúng 5 cặp QA theo PHÂN BỔ LOẠI sau (BẮT BUỘC tuân thủ):

1. factoid (1 câu): Hỏi thông tin cụ thể — ai, cái gì, khi nào, ở đâu, con số.
   - Câu trả lời ngắn gọn, trích xuất trực tiếp từ bài.
   - Reasoning: thấp.

2. event_summary (1 câu): Yêu cầu tóm tắt sự kiện chính.
   - Câu trả lời là 1-2 câu tóm tắt nội dung cốt lõi.
   - Reasoning: trung bình.

3. cause_effect HOẶC entity_role HOẶC comparison (2 câu, chọn 2 trong 3 loại):
   - cause_effect: Hỏi nguyên nhân / hệ quả của sự kiện trong bài.
   - entity_role: Hỏi vai trò cụ thể của cá nhân/tổ chức được nhắc trong bài.
   - comparison: So sánh 2 đối tượng/quan điểm/số liệu trong cùng bài.
   - Reasoning: trung bình.

4. claim_verification HOẶC unanswerable (1 câu, chọn 1 trong 2):
   - claim_verification: Đưa ra một nhận định và hỏi đúng/sai dựa trên bài. Câu trả lời phải có dẫn chứng.
   - unanswerable: Câu hỏi mà bài báo KHÔNG đủ thông tin trả lời (hỏi cảm xúc, dự đoán, chi tiết không đề cập).
     Khi unanswerable: is_possible = false, answers = [], plausible_answers chứa 1-2 câu suy luận hợp lý.

Quy tắc output:
- Mỗi item PHẢI có trường "qa_type" ghi rõ loại: factoid | event_summary | cause_effect | entity_role | comparison | claim_verification | unanswerable.
- is_possible = true cho tất cả loại NGOẠI TRỪ unanswerable (is_possible = false).
- answers: danh sách 1-3 câu trả lời ngắn (trích dẫn hoặc diễn giải sát nội dung bài).
- plausible_answers: 1-2 câu trả lời có vẻ hợp lý nhưng KHÔNG được xác nhận rõ trong bài.
- Trả về CHỈ JSON array, không text thừa, không markdown code block.

Định dạng output:
[
  {
    "question": "...",
    "qa_type": "factoid",
    "answers": ["..."],
    "is_possible": true,
    "plausible_answers": ["..."]
  },
  {
    "question": "...",
    "qa_type": "unanswerable",
    "answers": [],
    "is_possible": false,
    "plausible_answers": ["...", "..."]
  },
  ...
]"""

USER_TEMPLATE = """Bài báo (ID: {article_id}):
Tiêu đề: {title}
Nội dung:
{content}

Sinh 5 câu hỏi QA cho bài báo này."""

# ── Cross-article prompts ──────────────────────────────────────────────────
CROSS_ARTICLE_SYSTEM_PROMPT = """Bạn là chuyên gia tạo câu hỏi tổng hợp (cross-article QA) từ NHIỀU bài báo tiếng Việt, phục vụ đánh giá hệ thống RAG.

Bạn sẽ được cung cấp nhiều bài báo cùng lúc. Nhiệm vụ: tạo câu hỏi thuộc 2 loại sau:

1. multi_doc_comparison (bắt buộc ít nhất 1 câu):
   - So sánh, đối chiếu thông tin giữa 2+ bài báo.
   - CẦN đọc nhiều bài mới trả lời được, KHÔNG tạo câu chỉ cần 1 bài.
   - Ví dụ: "Điểm chung/khác biệt giữa sự kiện ở bài 1 và bài 2?"
   - Reasoning: cao.

2. timeline (nếu các bài có yếu tố thời gian):
   - Sắp xếp/tổng hợp diễn biến theo thời gian từ nhiều bài.
   - Ví dụ: "Trình tự thời gian các sự kiện từ bài 1, 2, 3 diễn ra như thế nào?"
   - Reasoning: cao.

Phân bổ is_possible:
- Khoảng một nửa câu is_possible = true (ghép thông tin từ các bài thì trả lời được).
- Khoảng một nửa câu is_possible = false (các bài KHÔNG đủ thông tin, cần suy luận ngoài).
- Khi false: answers = [], plausible_answers chứa 1-2 câu suy luận hợp lý.

Quy tắc output:
- Mỗi item PHẢI có trường "qa_type": multi_doc_comparison | timeline.
- Trả về CHỈ JSON array, không text thừa, không markdown code block.

Định dạng output:
[
  {
    "question": "...",
    "qa_type": "multi_doc_comparison",
    "answers": ["..."],
    "is_possible": true,
    "plausible_answers": ["..."],
    "source_article_ids": [id1, id2]
  }
]"""

CROSS_ARTICLE_USER_TEMPLATE = """Dưới đây là {num_articles} bài báo:

{articles_text}

Hãy tạo {num_questions} câu hỏi tổng hợp (cross-article) cần thông tin từ NHIỀU bài báo trên mới trả lời được."""


def build_cross_article_groups(df: pd.DataFrame) -> list[tuple]:
    """Gom nhóm bài báo theo category + AgglomerativeClustering trên TF-IDF.

    Pipeline:
    1. groupby('category')
    2. TF-IDF trên (title + description + 500 ký tự content đầu)
    3. AgglomerativeClustering với distance_threshold tự chia cluster
    4. Trong mỗi cluster, chia sub-group đúng CROSS_ARTICLE_GROUP_SIZE
       bằng top-k similarity >= MIN_SIMILARITY
    5. Bài thừa / cluster nhỏ bị bỏ qua
    """
    groups: list[tuple] = []
    gs = CROSS_ARTICLE_GROUP_SIZE

    for category, group_df in df.groupby('category'):
        cat_groups: list[tuple] = []  # [(avg_sim, (category, articles))]
        all_articles = group_df.to_dict('records')
        total_in_cat = len(all_articles)

        if total_in_cat < gs:
            print(f"  [{category}] {total_in_cat} bài < {gs}, bỏ qua")
            continue

        # ── Sample nhỏ để clustering nhanh ──────────────────────────────
        if total_in_cat > SAMPLE_PER_CATEGORY:
            import random
            articles = random.sample(all_articles, SAMPLE_PER_CATEGORY)
        else:
            articles = all_articles
        n = len(articles)

        # ── Bước 1: TF-IDF ──────────────────────────────────────────────
        texts = []
        for art in articles:
            title = str(art.get('title', ''))
            desc  = str(art.get('description', ''))
            cont  = str(art.get('content', ''))[:500]
            texts.append(f"{title} {desc} {cont}")

        try:
            vectorizer = TfidfVectorizer(max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(texts)
            sim_matrix = cosine_similarity(tfidf_matrix)
        except Exception as e:
            print(f"  [WARN] TF-IDF lỗi cho '{category}': {e}, gom tuần tự")
            for i in range(0, min(n, gs * MAX_CROSS_GROUPS_PER_CATEGORY), gs):
                grp = articles[i:i + gs]
                if len(grp) == gs:
                    groups.append((category, grp))
            continue

        # ── Bước 2: Top-k greedy (nhanh, đủ tốt cho sample nhỏ) ────────
        sub_groups = _split_cluster_topk(list(range(n)), sim_matrix, group_size=gs)
        for sg in sub_groups:
            grp = [articles[i] for i in sg]
            avg_sim = _avg_pairwise_sim(sim_matrix, sg)
            cat_groups.append((avg_sim, (category, grp)))

        # Sort theo similarity giảm dần, chỉ giữ top N nhóm tốt nhất
        cat_groups.sort(key=lambda x: x[0], reverse=True)
        top_groups = cat_groups[:MAX_CROSS_GROUPS_PER_CATEGORY]
        groups.extend([g for _, g in top_groups])
        kept = len(top_groups)
        if top_groups:
            print(f"  [{category}] {total_in_cat} bài (sample {n}) → {len(cat_groups)} nhóm, giữ top {kept} (sim: {', '.join(f'{s:.3f}' for s, _ in top_groups)})")
        else:
            print(f"  [{category}] {total_in_cat} bài (sample {n}) → 0 nhóm đạt chuẩn")

    return groups


def _avg_pairwise_sim(sim_matrix: np.ndarray, indices: list[int]) -> float:
    """Trung bình cosine similarity giữa các cặp trong nhóm."""
    if len(indices) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            total += sim_matrix[indices[i], indices[j]]
            count += 1
    return total / count if count > 0 else 0.0


def _split_cluster_topk(
    members: list[int],
    sim_matrix: np.ndarray,
    group_size: int
) -> list[list[int]]:
    """Chia cluster lớn thành sub-group bằng greedy top-k similarity.

    1. Chọn bài có avg sim cao nhất làm seed
    2. Chọn (group_size-1) bài gần seed nhất
    3. Kiểm tra avg_sim nhóm >= MIN_SIMILARITY
    4. Lặp lại
    """
    sub_groups: list[list[int]] = []
    used = set()
    remaining = list(members)

    while len(remaining) >= group_size:
        # Tìm seed: bài có avg similarity cao nhất với remaining
        best_seed = None
        best_avg = -1
        for m in remaining:
            if m in used:
                continue
            others = [o for o in remaining if o != m and o not in used]
            if not others:
                continue
            avg = float(np.mean([sim_matrix[m, o] for o in others]))
            if avg > best_avg:
                best_avg = avg
                best_seed = m

        if best_seed is None:
            break

        # Top-(group_size-1) bài gần seed nhất
        candidates = [o for o in remaining if o != best_seed and o not in used]
        candidates.sort(key=lambda o: sim_matrix[best_seed, o], reverse=True)
        chosen = [best_seed] + candidates[:group_size - 1]

        if len(chosen) < group_size:
            break

        # Kiểm tra chất lượng nhóm
        avg_sim = _avg_pairwise_sim(sim_matrix, chosen)
        if avg_sim >= MIN_SIMILARITY:
            sub_groups.append(chosen)

        used.update(chosen)
        remaining = [m for m in remaining if m not in used]

    return sub_groups


def load_chunk_dataset() -> pd.DataFrame:
    """Build the article table and chunk cache from the E5 token JSONL only."""
    ARTICLE_CHUNKS.clear()
    article_rows: dict[str, dict] = {}

    with open(CHUNKS_JSONL, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Chunk JSON không hợp lệ tại dòng {line_number}: {exc}"
                ) from exc

            metadata = chunk.get("metadata", {})
            article_id = normalize_article_id(
                chunk.get("article_id") or metadata.get("article_id")
            )
            chunk_id = str(chunk.get("chunk_id", "")).strip()
            chunk_text = str(chunk.get("chunk_text") or chunk.get("text") or "").strip()
            if not article_id or not chunk_id or not chunk_text:
                continue

            ARTICLE_CHUNKS.setdefault(article_id, []).append({
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
            })
            if article_id not in article_rows:
                article_rows[article_id] = {
                    "id": int(article_id) if article_id.isdigit() else article_id,
                    "title": metadata.get("title") or chunk.get("title") or "",
                    "description": metadata.get("description") or "",
                    "category": metadata.get("category") or "unknown",
                    # Used only by TF-IDF grouping; QA context comes from ARTICLE_CHUNKS.
                    "content": chunk_text,
                    "num_chunks": int(metadata.get("num_chunks", 0) or 0),
                }

    for article_id, chunks in ARTICLE_CHUNKS.items():
        chunks.sort(key=lambda item: item["chunk_id"])
        article_rows[article_id]["num_chunks"] = len(chunks)

    df = pd.DataFrame(article_rows.values())
    print(f"Đầu vào E5 token: {len(df)} bài, {sum(map(len, ARTICLE_CHUNKS.values()))} chunk")
    return df


def load_article_chunks(article_ids) -> None:
    """Stream the token file and cache chunks only for requested articles."""
    requested = {
        normalize_article_id(article_id)
        for article_id in article_ids
        if normalize_article_id(article_id)
    }
    needed = requested - set(ARTICLE_CHUNKS)
    if not needed:
        return

    found = {article_id: [] for article_id in needed}
    with open(CHUNKS_JSONL, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Chunk JSON không hợp lệ tại dòng {line_number}: {exc}"
                ) from exc

            article_id = normalize_article_id(
                chunk.get("article_id") or chunk.get("metadata", {}).get("article_id")
            )
            if article_id not in needed:
                continue
            chunk_id = str(chunk.get("chunk_id", "")).strip()
            chunk_text = str(chunk.get("chunk_text") or chunk.get("text") or "").strip()
            if chunk_id and chunk_text:
                found[article_id].append({
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_text,
                })

    for article_id in needed:
        found[article_id].sort(key=lambda item: item["chunk_id"])
        ARTICLE_CHUNKS[article_id] = found[article_id]

    missing = sum(not ARTICLE_CHUNKS[article_id] for article_id in needed)
    loaded_chunks = sum(len(ARTICLE_CHUNKS[article_id]) for article_id in needed)
    print(
        f"  Chunk context: nạp {loaded_chunks} chunk cho {len(needed) - missing}/{len(needed)} bài"
    )
    if missing:
        print(f"  [WARN] {missing} bài không có token chunk, dùng toàn bộ content từ CSV")


def select_evenly_spaced_chunks(chunks: list[dict], limit: int) -> list[dict]:
    """Keep all short articles; sample long articles evenly from start to end."""
    if len(chunks) <= limit:
        return chunks
    if limit <= 1:
        return [chunks[0]]
    indices = {
        round(position * (len(chunks) - 1) / (limit - 1))
        for position in range(limit)
    }
    return [chunks[index] for index in sorted(indices)]


def build_chunk_context(article_id, fallback_content, max_chunks: int) -> str:
    """Build an LLM context from token chunks spanning the whole article."""
    normalized_id = normalize_article_id(article_id)
    all_chunks = ARTICLE_CHUNKS.get(normalized_id, [])
    selected = select_evenly_spaced_chunks(all_chunks, max_chunks)
    if not selected:
        return str(fallback_content or "")

    total_chunks = len(all_chunks)
    sections = []
    for chunk in selected:
        sections.append(
            f"[CHUNK_ID: {chunk['chunk_id']} | tổng số chunk của bài: {total_chunks}]\n"
            f"{chunk['chunk_text']}"
        )
    return "\n\n".join(sections)


def build_chunk_schedule(chunks: list[dict], slots: int) -> list[str]:
    """Assign QA slots to chunk positions spread across the selected context."""
    if not chunks or slots <= 0:
        return []
    if slots == 1:
        return [chunks[len(chunks) // 2]["chunk_id"]]
    return [
        chunks[round(slot * (len(chunks) - 1) / (slots - 1))]["chunk_id"]
        for slot in range(slots)
    ]


def keep_valid_chunk_hints(qa: dict, allowed_chunk_ids: set[str]) -> list[str]:
    """Keep only source hints that were actually present in the LLM context."""
    raw_hints = qa.get("source_chunk_ids", [])
    if not isinstance(raw_hints, list):
        raw_hints = [raw_hints]
    result = []
    for value in raw_hints:
        chunk_id = str(value).strip()
        if chunk_id in allowed_chunk_ids and chunk_id not in result:
            result.append(chunk_id)
    return result


def generate_cross_article_qa(articles: list[dict], num_questions: int = 2) -> list[dict]:
    """Tạo câu hỏi cross-article từ một nhóm bài báo."""
    articles_text = ""
    selected_by_article: dict[str, list[dict]] = {}
    for art in articles:
        article_id = normalize_article_id(art["id"])
        selected_by_article[article_id] = select_evenly_spaced_chunks(
            ARTICLE_CHUNKS.get(article_id, []),
            MAX_CROSS_CONTEXT_CHUNKS_PER_ARTICLE,
        )
        content_context = build_chunk_context(
            art["id"],
            art.get("content", ""),
            MAX_CROSS_CONTEXT_CHUNKS_PER_ARTICLE,
        )
        title_str = str(art.get("title", ""))
        articles_text += f"--- Bài báo (ID: {art['id']}) ---\nTiêu đề: {title_str}\nNội dung theo chunk:\n{content_context}\n\n"

    per_article_schedules = {
        article_id: build_chunk_schedule(chunks, num_questions)
        for article_id, chunks in selected_by_article.items()
    }
    question_plans = []
    for question_index in range(num_questions):
        chunk_ids = [
            schedule[question_index]
            for schedule in per_article_schedules.values()
            if len(schedule) > question_index
        ]
        question_plans.append(chunk_ids)
    plan_lines = "\n".join(
        f"- Cross QA số {index + 1}: source_chunk_ids={json.dumps(chunk_ids, ensure_ascii=False)}"
        for index, chunk_ids in enumerate(question_plans)
    )
    source_plan_prompt = (
        CROSS_DIVERSITY_PROMPT
        + "\nMỗi JSON item BẮT BUỘC có trường source_chunk_ids. "
        "Câu có đáp án phải dùng đúng một chunk từ mỗi bài theo kế hoạch; "
        "câu không có đáp án dùng source_chunk_ids=[].\n"
        + plan_lines
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": CROSS_ARTICLE_SYSTEM_PROMPT},
            {"role": "system", "content": source_plan_prompt},
            {"role": "user", "content": CROSS_ARTICLE_USER_TEMPLATE.format(
                num_articles=len(articles),
                articles_text=articles_text,
                num_questions=num_questions
            )}
        ]
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    qa_list = json.loads(raw)
    allowed_by_article = {
        article_id: {chunk["chunk_id"] for chunk in chunks}
        for article_id, chunks in selected_by_article.items()
    }
    all_allowed = set().union(*allowed_by_article.values()) if allowed_by_article else set()
    for qa in qa_list:
        if qa.get("is_possible") is False or not qa.get("answers"):
            qa["source_chunk_ids"] = []
            continue
        valid_hints = keep_valid_chunk_hints(qa, all_allowed)
        covers_all_articles = all(
            any(chunk_id in allowed_ids for chunk_id in valid_hints)
            for allowed_ids in allowed_by_article.values()
        )
        qa["source_chunk_ids"] = valid_hints if covers_all_articles else []
    return qa_list


def process_cross_article_group(articles: list[dict]) -> list[dict]:
    """Xử lý một nhóm bài báo để tạo cross-article QA."""
    qa_list = generate_cross_article_qa(articles, num_questions=CROSS_ARTICLE_QA_COUNT)
    article_ids = [art["id"] for art in articles]
    group_key = "_".join(sorted(normalize_article_id(article_id) for article_id in article_ids))

    records = []
    for i, qa in enumerate(qa_list):
        qa_type = qa.get("qa_type", "multi_doc_comparison")
        if qa_type not in ("multi_doc_comparison", "timeline"):
            qa_type = "multi_doc_comparison"
        records.append({
            "id": f"cross_{group_key}_{i+1}",
            "article_id": article_ids,
            "question": qa.get("question", ""),
            "qa_type": qa_type,
            "answers": qa.get("answers", []),
            "is_possible": qa.get("is_possible", True),
            "plausible_answers": qa.get("plausible_answers", []),
            "source_article_ids": qa.get("source_article_ids", article_ids),
            "source_chunk_ids": qa.get("source_chunk_ids", [])
        })
    return records


def validate_qa_taxonomy(qa_list: list[dict]) -> list[dict]:
    """Post-processing: validate và bổ sung qa_type nếu thiếu.

    Đảm bảo:
    - Mỗi item có trường qa_type hợp lệ
    - Có ít nhất 1 câu unanswerable/claim_verification (is_possible=false)
    - Nếu LLM không tuân thủ phân bổ, gán qa_type dựa trên heuristic
    """
    VALID_SINGLE_TYPES = {"factoid", "event_summary", "cause_effect", "entity_role",
                          "comparison", "claim_verification", "unanswerable"}

    for qa in qa_list:
        # Gán qa_type nếu thiếu hoặc không hợp lệ
        qa_type = qa.get("qa_type", "").strip().lower()
        if qa_type not in VALID_SINGLE_TYPES:
            # Heuristic: nếu is_possible=false → unanswerable, ngược lại → factoid
            if qa.get("is_possible") is False:
                qa["qa_type"] = "unanswerable"
            else:
                qa["qa_type"] = "factoid"
        else:
            qa["qa_type"] = qa_type

        # Đảm bảo consistency giữa qa_type và is_possible
        if qa["qa_type"] == "unanswerable":
            qa["is_possible"] = False
            if qa.get("answers"):
                # Chuyển answers sang plausible_answers
                existing_plausible = qa.get("plausible_answers", [])
                qa["plausible_answers"] = (existing_plausible or qa["answers"])[:2]
                qa["answers"] = []
        elif qa["qa_type"] != "unanswerable":
            qa["is_possible"] = True

    # Kiểm tra có ít nhất 1 câu unanswerable/claim_verification
    has_hard = any(q["qa_type"] in ("unanswerable", "claim_verification") for q in qa_list)
    if not has_hard and len(qa_list) >= 5:
        # Chuyển câu cuối thành unanswerable
        last = qa_list[-1]
        last["qa_type"] = "unanswerable"
        last["is_possible"] = False
        old_answers = last.get("answers", [])
        last["plausible_answers"] = (last.get("plausible_answers") or old_answers)[:2]
        last["answers"] = []

    return qa_list


def generate_qa(article_id: int, title: str, content: str) -> list[dict]:
    """Gọi LLM API để sinh QA cho một bài báo."""
    all_chunks = ARTICLE_CHUNKS.get(normalize_article_id(article_id), [])
    selected_chunks = select_evenly_spaced_chunks(
        all_chunks, MAX_SINGLE_CONTEXT_CHUNKS
    )
    content_context = build_chunk_context(
        article_id,
        content,
        MAX_SINGLE_CONTEXT_CHUNKS,
    )
    chunk_schedule = build_chunk_schedule(selected_chunks, QUESTIONS_PER_ARTICLE)
    schedule_lines = "\n".join(
        f"- QA số {index + 1}: dùng bằng chứng chính từ {chunk_id} và trả "
        f"source_chunk_ids=[\"{chunk_id}\"]"
        for index, chunk_id in enumerate(chunk_schedule)
    )
    source_plan_prompt = (
        SINGLE_DIVERSITY_PROMPT
        + "\nMỗi JSON item BẮT BUỘC có trường source_chunk_ids. "
        "Câu unanswerable dùng source_chunk_ids=[].\n"
        + schedule_lines
    )
    title_str = str(title) if title else ""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": source_plan_prompt},
            {"role": "user", "content": USER_TEMPLATE.format(
                article_id=article_id,
                title=title_str,
                content=content_context
            )}
        ]
    )

    raw = response.choices[0].message.content.strip()

    # Tách JSON array từ response (đề phòng có markdown code block)
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    qa_list = json.loads(raw)

    # Post-processing: validate taxonomy và consistency
    qa_list = validate_qa_taxonomy(qa_list)

    allowed_chunk_ids = {chunk["chunk_id"] for chunk in selected_chunks}
    for qa in qa_list:
        if qa.get("is_possible") is False or not qa.get("answers"):
            qa["source_chunk_ids"] = []
        else:
            qa["source_chunk_ids"] = keep_valid_chunk_hints(
                qa, allowed_chunk_ids
            )

    return qa_list


def process_article(row) -> list[dict]:
    """Xử lý một bài báo và trả về list record đã thêm id."""
    article_id = row["id"]
    qa_list = generate_qa(
        article_id=article_id,
        title=row.get("title", ""),
        content=row.get("content", "")
    )
    records = []
    for i, qa in enumerate(qa_list):
        records.append({
            "id": f"{article_id}_{i+1}",
            "article_id": article_id,
            "question": qa.get("question", ""),
            "qa_type": qa.get("qa_type", "factoid"),
            "answers": qa.get("answers", []),
            "is_possible": qa.get("is_possible", True),
            "plausible_answers": qa.get("plausible_answers", []),
            "source_chunk_ids": qa.get("source_chunk_ids", [])
        })
    return records


def normalize_article_id(value) -> str:
    """Chuẩn hóa ID để so sánh ổn định giữa CSV và JSONL."""
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def cross_group_signature(article_ids) -> tuple[str, ...]:
    """Tạo chữ ký không phụ thuộc thứ tự cho một nhóm cross-article."""
    return tuple(sorted(
        article_id
        for article_id in (normalize_article_id(value) for value in article_ids)
        if article_id
    ))


def load_processed_state(output_paths: list[str]) -> tuple[set[str], set[tuple[str, ...]], set[str]]:
    """Đọc ID bài đơn và nhóm cross-article đã có từ nhiều file JSONL."""
    single_ids: set[str] = set()
    cross_groups: set[tuple[str, ...]] = set()
    cross_article_ids: set[str] = set()
    seen_paths = set()

    for output_path in output_paths:
        normalized_path = os.path.abspath(output_path)
        if normalized_path in seen_paths or not os.path.exists(normalized_path):
            continue
        seen_paths.add(normalized_path)

        with open(normalized_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    article_id = rec.get("article_id")
                    if isinstance(article_id, list):
                        signature = cross_group_signature(article_id)
                        if signature:
                            cross_groups.add(signature)
                            cross_article_ids.update(signature)
                    else:
                        normalized_id = normalize_article_id(article_id)
                        if normalized_id:
                            single_ids.add(normalized_id)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue

    return single_ids, cross_groups, cross_article_ids


def main():
    df_source = load_chunk_dataset()
    processed_ids, processed_cross_groups, processed_cross_article_ids = load_processed_state([
        EXISTING_QA_JSONL,
        OUTPUT_JSONL,
    ])

    # Bắt đầu sau 20 dòng đầu, bỏ ID trùng rồi lấy đủ tối đa 60 bài kế tiếp.
    candidates = df_source.iloc[PHASE1_SKIP_ROWS:].drop_duplicates(subset=["id"], keep="first").copy()
    candidate_ids = candidates["id"].map(normalize_article_id)
    candidates = candidates.loc[~candidate_ids.isin(processed_ids)]
    diverse_candidates = candidates.loc[
        candidates["num_chunks"] >= MIN_SINGLE_CHUNKS_FOR_DIVERSITY
    ]
    fallback_candidates = candidates.loc[
        candidates["num_chunks"] < MIN_SINGLE_CHUNKS_FOR_DIVERSITY
    ]
    df = pd.concat([diverse_candidates, fallback_candidates], ignore_index=True).head(
        PHASE1_ARTICLE_COUNT
    ).copy()
    selected_phase1_ids = set(df["id"].map(normalize_article_id))
    load_article_chunks(selected_phase1_ids)

    total = len(df)
    print(f"Phase 1: bỏ {PHASE1_SKIP_ROWS} bài đầu, cần lấy {PHASE1_ARTICLE_COUNT} bài chưa xử lý")
    print(f"Đã tìm thấy {total} bài để xử lý; lịch sử có {len(processed_ids)} article_id")
    print(
        f"  Ưu tiên bài >= {MIN_SINGLE_CHUNKS_FOR_DIVERSITY} chunk: "
        f"{sum(df['num_chunks'] >= MIN_SINGLE_CHUNKS_FOR_DIVERSITY)}/{total} bài"
    )
    if total < PHASE1_ARTICLE_COUNT:
        print(f"  [WARN] Chỉ còn {total} bài chưa trùng trong CSV, không đủ {PHASE1_ARTICLE_COUNT} bài")

    out_file = open(OUTPUT_JSONL, "a", encoding="utf-8")
    buffer = []

    # ── Phase 1: Single-article QA ──────────────────────────────────────
    print("\n=== Phase 1: Single-article QA ===")
    try:
        for idx, (_, row) in enumerate(df.iterrows()):
            article_id = row["id"]
            try:
                records = process_article(row)
                buffer.extend(records)

                progress = idx + 1
                print(f"[{progress}/{total}] ID={article_id} → {len(records)} QA")

            except json.JSONDecodeError as e:
                print(f"  [WARN] ID={article_id} JSON parse error: {e}")
            except openai.RateLimitError:
                print(f"  [WARN] Rate limit, chờ 60s...")
                time.sleep(60)
                # Thử lại một lần
                try:
                    records = process_article(row)
                    buffer.extend(records)
                except Exception as e2:
                    print(f"  [ERROR] Bỏ qua ID={article_id}: {e2}")
            except Exception as e:
                print(f"  [ERROR] ID={article_id}: {e}")

            # Flush ra file mỗi BATCH_SIZE bài
            if len(buffer) >= BATCH_SIZE * QUESTIONS_PER_ARTICLE:
                for rec in buffer:
                    out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_file.flush()
                buffer.clear()

            # Tránh rate limit
            time.sleep(0.5)

    finally:
        # Flush phần còn lại single-article
        for rec in buffer:
            out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_file.flush()
        buffer.clear()

        # ── Phase 2: Cross-article QA ───────────────────────────────────────
        print("\n=== Phase 2: Cross-article QA ===")
        # Không dùng lại bài đã xuất trước đây hoặc đã chọn cho Phase 1 lần này.
        excluded_phase2_ids = processed_ids | processed_cross_article_ids | selected_phase1_ids
        df_all = df_source.drop_duplicates(subset=["id"], keep="first").copy()
        df_all_ids = df_all["id"].map(normalize_article_id)
        df_cross_pool = df_all.loc[~df_all_ids.isin(excluded_phase2_ids)].copy()
        print(
            f"  Nguồn Phase 2: {len(df_cross_pool)}/{len(df_all)} bài sau khi loại "
            f"{len(excluded_phase2_ids)} ID đã dùng"
        )
        candidate_groups = build_cross_article_groups(df_cross_pool)

        # Chặn nhóm đã xuất ở lần chạy trước và chặn trùng nhóm trong cùng lần chạy.
        groups = []
        for category, group in candidate_groups:
            signature = cross_group_signature(art["id"] for art in group)
            if not signature or signature in processed_cross_groups:
                continue
            processed_cross_groups.add(signature)
            groups.append((category, group))

        selected_cross_ids = {
            normalize_article_id(article["id"])
            for _, group in groups
            for article in group
        }
        load_article_chunks(selected_cross_ids)

    num_groups = len(groups)

    print(f"Tổng {num_groups} nhóm cross-article (gom theo category, mỗi nhóm {CROSS_ARTICLE_GROUP_SIZE} bài, {CROSS_ARTICLE_QA_COUNT} câu hỏi)")

    try:
        for g, (category, group) in enumerate(groups):
            group_ids = [a["id"] for a in group]

            try:
                records = process_cross_article_group(group)
                buffer.extend(records)
                print(f"  [Cross {g+1}/{num_groups}] Category='{category}' IDs={group_ids} → {len(records)} QA")
            except json.JSONDecodeError as e:
                print(f"  [WARN] Cross group {g} JSON parse error: {e}")
            except openai.RateLimitError:
                print(f"  [WARN] Rate limit, chờ 60s...")
                time.sleep(60)
                try:
                    records = process_cross_article_group(group)
                    buffer.extend(records)
                except Exception as e2:
                    print(f"  [ERROR] Bỏ qua cross group {g}: {e2}")
            except Exception as e:
                print(f"  [ERROR] Cross group {g}: {e}")

            if len(buffer) >= BATCH_SIZE * CROSS_ARTICLE_QA_COUNT:
                for rec in buffer:
                    out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_file.flush()
                buffer.clear()

            time.sleep(0.5)
    finally:
        for rec in buffer:
            out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_file.close()
        print("Hoàn tất. Kết quả lưu tại:", OUTPUT_JSONL)


if __name__ == "__main__":
    main()
