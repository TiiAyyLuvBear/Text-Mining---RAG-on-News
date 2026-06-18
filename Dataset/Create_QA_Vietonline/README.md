# Create_QA_Vietonline

Bộ script tạo dataset QA (Question - Answer) tiếng Việt từ kho tin tức **VietOnlineNews**. Pipeline gồm 3 bước: tải dữ liệu → sinh QA bằng LLM → chuyển sang CSV.

## Cấu trúc thư mục

| File | Vai trò |
|------|---------|
| `data.py` | Tải dataset `VLUS06/VietOnlineNews` từ Hugging Face và xuất ra CSV |
| `QA.py` | Sinh cặp QA (single-article + cross-article) bằng LLM API, lưu ra JSONL |
| `CONVERT_QA.py` | Chuyển file JSONL kết quả sang CSV |

Sau khi chạy, thư mục sẽ phát sinh thêm:

```
VietOnlineNews_CSV/      # data.py tạo ra (train.csv, validation.csv, test.csv)
QA_output.jsonl          # QA.py tạo ra
QA_output.csv            # CONVERT_QA.py tạo ra
```

## Yêu cầu cài đặt

```bash
pip install datasets pandas openai scikit-learn numpy
```

## Hướng dẫn dùng

### Bước 1 — Tải dữ liệu (`data.py`)

```bash
python data.py
```

Tải dataset từ Hugging Face và lưu từng split (`train`, `validation`, `test`) thành CSV trong `VietOnlineNews_CSV/`. File `train.csv` là input cho bước sau.

### Bước 2 — Sinh QA (`QA.py`)

Mở `QA.py` và chỉnh phần **Config** ở đầu file trước khi chạy:

| Tham số | Ý nghĩa |
|---------|---------|
| `API_KEY` | Khóa API của bạn (bắt buộc thay) |
| `BASE_URL` | Endpoint API tương thích OpenAI |
| `MODEL` | Tên model dùng để sinh QA |
| `INPUT_CSV` | File CSV đầu vào (mặc định `VietOnlineNews_CSV/train.csv`) |
| `OUTPUT_JSONL` | File JSONL kết quả |
| `MAX_ROWS` | Số bài xử lý (đặt số nhỏ để test, `None` = toàn bộ) |
| `QUESTIONS_PER_ARTICLE` | Số câu hỏi mỗi bài (single-article) |
| `CROSS_ARTICLE_GROUP_SIZE` | Số bài gom nhóm cho câu hỏi cross-article |
| `CROSS_ARTICLE_QA_COUNT` | Số câu hỏi mỗi nhóm cross-article |
| `MIN_SIMILARITY` | Ngưỡng similarity tối thiểu để gom nhóm |

Chạy:

```bash
python QA.py
```

Quy trình gồm 2 phase:

- **Phase 1 — Single-article QA:** mỗi bài sinh `QUESTIONS_PER_ARTICLE` câu hỏi, có cân bằng tỉ lệ câu trả lời được / không trả lời được (`is_possible`).
- **Phase 2 — Cross-article QA:** gom bài cùng category bằng TF-IDF + clustering, tạo câu hỏi cần tổng hợp từ nhiều bài.

Lưu ý: script hỗ trợ **resume** — nếu bị ngắt giữa chừng, chạy lại sẽ bỏ qua các bài đã có trong `QA_output.jsonl`.

### Bước 3 — Chuyển sang CSV (`CONVERT_QA.py`)

```bash
python CONVERT_QA.py
```

Đọc `QA_output.jsonl` và xuất `QA_output.csv` (mã hóa `utf-8-sig` để mở tốt trong Excel).

## Định dạng output

Mỗi dòng JSONL là một cặp QA:

```json
{
  "id": "123_1",
  "article_id": 123,
  "question": "...",
  "answers": ["..."],
  "is_possible": true,
  "plausible_answers": ["..."]
}
```

- `is_possible = true`: bài báo có đủ thông tin để trả lời → `answers` chứa đáp án.
- `is_possible = false`: thông tin không có trong bài → `answers` rỗng, `plausible_answers` chứa đáp án suy luận.
- Câu hỏi cross-article có thêm `source_article_ids` và `qa_type = "cross-article"`, `id` dạng `cross_<nhóm>_<số>`.
