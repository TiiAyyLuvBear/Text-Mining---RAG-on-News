import pandas as pd
import openai
import json
import os
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_CSV      = "VietOnlineNews_CSV/train.csv"
OUTPUT_JSONL   = "QA_output.jsonl"

# API config
API_KEY   = "................"
BASE_URL  = "https://api.xah.io/v1"   # <-- SỬA LẠI BASE_URL NẾU CẦN
MODEL     = "lohieuky1/Claude Opus 4.8"

BATCH_SIZE  = 5
MAX_ROWS    = 20        # None = toàn bộ, đặt số để test (ví dụ 10)
QUESTIONS_PER_ARTICLE = 5
CROSS_ARTICLE_GROUP_SIZE = 3   # Số bài báo gom lại để tạo câu hỏi cross-article
CROSS_ARTICLE_QA_COUNT = 2    # Số câu hỏi cross-article mỗi nhóm
MIN_SIMILARITY = 0.15         # Ngưỡng similarity tối thiểu để nhóm bài lại
MAX_CROSS_GROUPS_PER_CATEGORY = 2  # Mỗi category chỉ lấy tối đa N nhóm tốt nhất
SAMPLE_PER_CATEGORY = 50           # Mỗi category chỉ sample N bài để clustering (tránh chậm)

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = """Bạn là chuyên gia tạo bộ câu hỏi – câu trả lời (QA) từ bài báo tiếng Việt.
Với mỗi bài báo được cung cấp, hãy sinh đúng 5 cặp QA theo định dạng JSON.

Quy tắc QUAN TRỌNG về phân bổ is_possible:
- Trong 5 câu hỏi, PHẢI có từ 2 đến 3 câu có is_possible = true và 2 đến 3 câu có is_possible = false.
- Cụ thể: tuyệt đối KHÔNG được tạo 5 câu true hoặc 4 câu true. Phân bổ lý tưởng là 3 true + 2 false hoặc 2 true + 3 false.
- is_possible = true: nội dung bài báo chứa đủ thông tin để trả lời chính xác.
- is_possible = false: câu hỏi yêu cầu thông tin KHÔNG có trong bài báo (ví dụ: hỏi về bối cảnh bên ngoài, cảm xúc cá nhân, chi tiết chưa được nhắc đến).
- Để tạo câu false, hãy đặt câu hỏi về khía cạnh bài báo KHÔNG đề cập: cảm xúc nhân vật, chi tiết cá nhân, dự đoán tương lai, lý do ẩn ý, thông tin bên lề...
- Khi tạo câu false, answers phải là rỗng [], và plausible_answers chứa 1-2 câu trả lời suy luận hợp lý.

Quy tắc khác:
- Câu hỏi phải đa dạng: ai, cái gì, khi nào, ở đâu, tại sao, như thế nào.
- answers: danh sách 1-3 câu trả lời ngắn (trích dẫn hoặc diễn giải từ nội dung).
- plausible_answers: 1-2 câu trả lời hợp lý nhưng KHÔNG được xác nhận rõ trong bài (có thể suy luận).
- Trả về CHỈ JSON array, không có text thừa, không markdown.

VÍ DỤ về câu false:
- "Cảm xúc của nhân vật A khi xảy ra sự kiện B là gì?" → false vì bài không mô tả cảm xúc
- "Tại sao nhân vật A lại quyết định như vậy?" → false nếu bài chỉ nêu hành động mà không giải thích lý do
- "Kết quả cuối cùng của sự việc này là gì?" → false nếu bài viết trước khi sự việc kết thúc

Định dạng output:
[
  {
    "question": "...",
    "answers": ["..."],
    "is_possible": true,
    "plausible_answers": ["..."]
  },
  ...
]"""

USER_TEMPLATE = """Bài báo (ID: {article_id}):
Tiêu đề: {title}
Nội dung:
{content}

Sinh 5 câu hỏi QA cho bài báo này."""

# ── Cross-article prompts ──────────────────────────────────────────────────
CROSS_ARTICLE_SYSTEM_PROMPT = """Bạn là chuyên gia tạo câu hỏi tổng hợp (cross-article QA) từ NHIỀU bài báo tiếng Việt.
Bạn sẽ được cung cấp nhiều bài báo cùng lúc. Nhiệm vụ của bạn là tạo câu hỏi mà:
- CẦN PHẢI đọc/tổng hợp thông tin từ ÍT NHẤT 2 bài báo trở lên mới trả lời được.
- Câu hỏi so sánh, đối chiếu, tổng hợp xu hướng, tìm mối liên hệ giữa các bài.
- KHÔNG tạo câu hỏi chỉ cần 1 bài là trả lời được.

Phân bổ is_possible:
- Khoảng một nửa câu is_possible = true (thông tin nằm rải rác trong các bài, ghép lại thì trả lời được).
- Khoảng một nửa câu is_possible = false (câu hỏi tổng hợp nhưng các bài KHÔNG đủ thông tin để trả lời).
- Khi false: answers = [], plausible_answers chứa 1-2 câu trả lời suy luận hợp lý.

Ví dụ câu hỏi cross-article:
- "Điểm chung giữa sự kiện A (bài 1) và sự kiện B (bài 2) là gì?"
- "So sánh cách xử lý vấn đề X trong bài 1 và bài 2?"
- "Nếu kết hợp thông tin từ cả 3 bài, xu hướng chung là gì?"

Trả về CHỈ JSON array, không text thừa, không markdown.
Định dạng output:
[
  {
    "question": "...",
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


def generate_cross_article_qa(articles: list[dict], num_questions: int = 2) -> list[dict]:
    """Tạo câu hỏi cross-article từ một nhóm bài báo."""
    articles_text = ""
    for art in articles:
        content_trimmed = str(art["content"])[:2000] if art.get("content") else ""
        title_str = str(art.get("title", ""))
        articles_text += f"--- Bài báo (ID: {art['id']}) ---\nTiêu đề: {title_str}\nNội dung:\n{content_trimmed}\n\n"

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": CROSS_ARTICLE_SYSTEM_PROMPT},
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
    return qa_list


def process_cross_article_group(articles: list[dict], group_idx: int) -> list[dict]:
    """Xử lý một nhóm bài báo để tạo cross-article QA."""
    qa_list = generate_cross_article_qa(articles, num_questions=CROSS_ARTICLE_QA_COUNT)
    article_ids = [art["id"] for art in articles]

    records = []
    for i, qa in enumerate(qa_list):
        records.append({
            "id": f"cross_{group_idx}_{i+1}",
            "article_id": article_ids,
            "question": qa.get("question", ""),
            "answers": qa.get("answers", []),
            "is_possible": qa.get("is_possible", True),
            "plausible_answers": qa.get("plausible_answers", []),
            "source_article_ids": qa.get("source_article_ids", article_ids),
            "qa_type": "cross-article"
        })
    return records


def rebalance_qa(qa_list: list[dict], target_false: int = 2) -> list[dict]:
    """Post-processing: đảm bảo có đủ câu false trong danh sách.
    Nếu ít hơn target_false câu false, chuyển một số câu true thành false.
    """
    import random

    true_items = [i for i, q in enumerate(qa_list) if q.get("is_possible") is True]
    false_items = [i for i, q in enumerate(qa_list) if q.get("is_possible") is False]

    current_false = len(false_items)
    need_false = target_false - current_false

    if need_false <= 0:
        return qa_list

    # Chuyển bớt câu true thành false
    convert_indices = true_items[:need_false]
    for idx in convert_indices:
        qa_list[idx]["is_possible"] = False
        # Đưa answers vào plausible_answers nếu chưa có
        old_answers = qa_list[idx].get("answers", [])
        old_plausible = qa_list[idx].get("plausible_answers", [])
        combined = old_plausible if old_plausible else old_answers
        qa_list[idx]["plausible_answers"] = combined[:2]
        qa_list[idx]["answers"] = []

    return qa_list


def generate_qa(article_id: int, title: str, content: str) -> list[dict]:
    """Gọi LLM API để sinh QA cho một bài báo."""
    content_trimmed = str(content)[:3000] if content else ""
    title_str = str(title) if title else ""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                article_id=article_id,
                title=title_str,
                content=content_trimmed
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

    # Post-processing: đảm bảo phân bổ true/false cân bằng
    qa_list = rebalance_qa(qa_list, target_false=2)

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
            "answers": qa.get("answers", []),
            "is_possible": qa.get("is_possible", True),
            "plausible_answers": qa.get("plausible_answers", [])
        })
    return records


def load_processed_ids(output_path: str) -> set:
    """Đọc các article_id đã xử lý để resume nếu bị ngắt."""
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add(rec["article_id"])
            except Exception:
                pass
    return done


def main():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    if MAX_ROWS:
        df = df.head(MAX_ROWS)

    total = len(df)
    print(f"Tổng số bài: {total}")

    # Resume support: bỏ qua bài đã làm
    processed_ids = load_processed_ids(OUTPUT_JSONL)
    if processed_ids:
        print(f"Resume: đã có {len(processed_ids)} bài, bỏ qua...")
        df = df[~df["id"].isin(processed_ids)]
    print(f"Còn lại cần xử lý: {len(df)} bài")

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

                progress = idx + 1 + len(processed_ids)
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
        df_all = pd.read_csv(INPUT_CSV, encoding="utf-8")
        # Phase 2 đọc TOÀN BỘ CSV, không bị ảnh hưởng MAX_ROWS
        print(f"  Đọc toàn bộ CSV: {len(df_all)} bài, {df_all['category'].nunique()} categories")
        groups = build_cross_article_groups(df_all)

    num_groups = len(groups)

    print(f"Tổng {num_groups} nhóm cross-article (gom theo category, mỗi nhóm {CROSS_ARTICLE_GROUP_SIZE} bài, {CROSS_ARTICLE_QA_COUNT} câu hỏi)")

    try:
        for g, (category, group) in enumerate(groups):
            group_ids = [a["id"] for a in group]

            try:
                records = process_cross_article_group(group, group_idx=g)
                buffer.extend(records)
                print(f"  [Cross {g+1}/{num_groups}] Category='{category}' IDs={group_ids} → {len(records)} QA")
            except json.JSONDecodeError as e:
                print(f"  [WARN] Cross group {g} JSON parse error: {e}")
            except openai.RateLimitError:
                print(f"  [WARN] Rate limit, chờ 60s...")
                time.sleep(60)
                try:
                    records = process_cross_article_group(group, group_idx=g)
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
