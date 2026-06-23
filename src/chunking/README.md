# Chunking Experiments

Module này tạo các biến thể chunking cho tập VieOnlineNews và ghi toàn bộ kết quả vào `src/chunking/output`.

## Strategies

| Strategy | Mục đích | Cách hoạt động |
| --- | --- | --- |
| `token` | Baseline | Cắt theo cửa sổ token cố định, có overlap token giữa hai chunk liên tiếp. |
| `langchain_recursive` | So sánh với LangChain | Dùng `RecursiveCharacterTextSplitter` nếu có `langchain-text-splitters`; nếu chưa cài thì dùng fallback recursive character splitter nội bộ. |
| `llamaindex` | So sánh với LlamaIndex | Dùng `llama_index.core.node_parser.SentenceSplitter`; nếu chưa cài thì dùng fallback sentence-window splitter nội bộ. |
| `structured` | Structure-based | Tôn trọng cấu trúc bài viết, đặc biệt là paragraph, rồi mới chia tiếp theo sentence/token window. |

Mỗi chunk vẫn giữ schema chung:

- `chunk_id`
- `article_id`
- `strategy`
- `text`: nội dung dùng cho embedding, mặc định có chèn title/description/category
- `chunk_text`: nội dung chunk gốc
- `metadata`: strategy, implementation, structure, vị trí ký tự, token count, số thứ tự chunk

## Chạy CLI

Chạy mặc định trên dữ liệu đã preprocess:

```powershell
python -m src.chunking.cli `
  --input data/processed/vieonline_news_clean.jsonl
```

Hoặc chạy trực tiếp file CLI:

```powershell
python src\chunking\cli.py `
  --input data\processed\vieonline_news_clean.jsonl
```

Mặc định CLI chạy cả 4 strategy:

```text
token langchain_recursive llamaindex structured
```

Output mặc định:

```text
src/chunking/output/
  vieonline_news_chunks_token.jsonl
  vieonline_news_chunks_langchain_recursive.jsonl
  vieonline_news_chunks_llamaindex.jsonl
  vieonline_news_chunks_structured.jsonl
  chunking_summary.json
  chunking_report.md
```

## Thử nghiệm nhanh trên subdataset

Subdataset nhỏ cho test nằm ở:

```text
tests/fixtures/chunking_subdataset.jsonl
```

Chạy thử:

```powershell
python -m src.chunking.cli `
  --input tests\fixtures\chunking_subdataset.jsonl `
  --output-dir src\chunking\output\subdataset `
  --chunk-size 45 `
  --overlap 8 `
  --small-article-chars 0
```

`--small-article-chars 0` buộc cả bài ngắn cũng phải đi qua strategy chunking, giúp dễ quan sát hành vi.

## Tham số chính

- `--chunk-size`: kích thước cửa sổ. Với `token` và `llamaindex`, giá trị này gần với token count; với `langchain_recursive`, đây là character window theo thiết kế của LangChain.
- `--overlap`: overlap giữa các chunk.
- `--min-chunk-tokens`: nếu chunk cuối quá ngắn, nó được gộp vào chunk trước.
- `--max-chunks-per-article`: giới hạn số chunk mỗi bài nếu cần debug nhanh.
- `--no-title-injection`: tắt việc chèn title/description/category vào trường `text`.
- `--limit`: chỉ đọc N bài đầu tiên từ input.
- `--no-progress`: tắt progress bar `tqdm`.

CLI hiển thị progress bar riêng cho từng strategy:

```text
Chunking [token]: 100%|██████████| 10073/10073 [00:03<00:00, ...article/s]
```

Sau mỗi strategy, CLI in thông báo hoàn tất gồm số article, số chunk, thời gian chạy và file output. Khi toàn bộ experiment kết thúc, CLI in đường dẫn `chunking_summary.json` và `chunking_report.md`.

## Phân tích kết quả

Sau mỗi lần chạy, CLI sinh `chunking_summary.json` và `chunking_report.md`.

Các chỉ số chính:

- `elapsed_seconds`: thời gian chunking của strategy.
- `chunks_written`: tổng số chunk.
- `avg_chunks_per_article`: số chunk trung bình trên mỗi bài.
- `avg_chunk_tokens`: độ dài trung bình theo token ước lượng.
- `min_chunk_tokens`, `max_chunk_tokens`: biên độ độ dài chunk.
- `chunks_per_second`: tốc độ xử lý.
- `truncated_articles`: số bài bị cắt bớt do `--max-chunks-per-article`.

Gợi ý đọc kết quả:

- `token` thường ổn định nhất để làm baseline vì kiểm soát trực tiếp chunk size và overlap.
- `langchain_recursive` thường giữ được ranh giới paragraph/câu tốt hơn khi văn bản có dấu xuống dòng rõ.
- `llamaindex` phù hợp khi muốn sentence-aware chunking nhất quán với stack LlamaIndex.
- `structured` nên được ưu tiên kiểm tra nếu retrieval cần giữ ngữ cảnh theo đoạn hoặc cấu trúc bài báo.
