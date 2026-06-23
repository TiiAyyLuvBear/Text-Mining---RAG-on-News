# Chunking Experiments

Module này tạo các biến thể chunking cho tập VieOnlineNews và ghi toàn bộ kết quả vào `src/chunking/output`.

## Strategies

| Strategy | Mục đích | Cách hoạt động |
| --- | --- | --- |
| `token` | Baseline ổn định | Cắt theo cửa sổ token ước lượng bằng regex `\S+`, mỗi chunk có tối đa `--chunk-size` token và bước nhảy là `chunk_size - overlap`. |
| `langchain_recursive` | So sánh với LangChain | Dùng `RecursiveCharacterTextSplitter` nếu có `langchain-text-splitters`; nếu chưa cài thì dùng fallback recursive character splitter nội bộ. Strategy này ưu tiên ranh giới `\n\n`, `\n`, câu, khoảng trắng rồi mới cắt nhỏ hơn. |
| `llamaindex` | Sentence-aware chunking | Dùng `llama_index.core.node_parser.SentenceSplitter` nếu có LlamaIndex; nếu chưa cài thì dùng fallback sentence-window splitter nội bộ. Strategy gom câu đến gần `--chunk-size`, sau đó overlap theo câu gần cuối. |
| `structured` | Structure-based | Tách nội dung thành paragraph trước, gắn `structure=paragraph_000`, `paragraph_001`, ... rồi chunk từng paragraph bằng sentence window. Nếu bài chỉ có một paragraph thì dùng `structure=content`. |

Trước khi áp dụng strategy, bài có `len(content) <= --small-article-chars` được giữ thành một chunk duy nhất với `implementation=single_small_article`. Điều này tránh chia quá nhỏ các bài ngắn. Muốn quan sát trực tiếp hành vi strategy trên bài ngắn, đặt `--small-article-chars 0`.

Sau khi chia, nếu chunk cuối có ít hơn `--min-chunk-tokens`, nó được gộp vào chunk trước bằng logic loại bỏ phần overlap trùng. Nếu bật `--max-chunks-per-article`, các chunk vượt giới hạn bị cắt bỏ và metadata `truncated_article=true`.

## Khi Nào Dùng Strategy Nào

- `token`: dùng làm baseline vì dễ kiểm soát độ dài và overlap, phù hợp để so sánh hiệu năng retrieval giữa các cấu hình.
- `langchain_recursive`: dùng khi muốn ưu tiên giữ paragraph/câu tự nhiên nhưng vẫn cần splitter phổ biến trong hệ sinh thái LangChain.
- `llamaindex`: dùng khi pipeline retrieval/indexing về sau dùng LlamaIndex hoặc muốn sentence-aware chunking nhất quán hơn cắt token thô.
- `structured`: dùng khi paragraph của bài báo có ý nghĩa rõ và cần metadata `structure` để phân tích chunk đến từ đoạn nào.

Mỗi chunk vẫn giữ schema chung:

- `chunk_id`
- `article_id`
- `strategy`
- `text`: nội dung dùng cho embedding, mặc định có chèn title/description/category
- `chunk_text`: nội dung chunk gốc
- `metadata`: strategy, implementation, structure, vị trí ký tự, token count, số thứ tự chunk

Ví dụ schema một chunk:

```json
{
  "chunk_id": "123_token_0000",
  "article_id": "123",
  "strategy": "token",
  "text": "Tiêu đề: ...\nMô tả: ...\nChuyên mục: ...\nĐoạn nội dung:\n...",
  "chunk_text": "Nội dung chunk gốc",
  "metadata": {
    "article_id": "123",
    "title": "Tiêu đề",
    "description": "Mô tả",
    "category": "Chuyên mục",
    "strategy": "token",
    "implementation": "internal_token_window",
    "structure": "content",
    "chunk_index": 0,
    "num_chunks": 4,
    "char_start": 0,
    "char_end": 1800,
    "token_count": 450,
    "chunk_chars": 1800,
    "truncated_article": false
  }
}
```

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

| Tham số | Mặc định | Ý nghĩa |
| --- | ---: | --- |
| `--input` | `data/processed/vieonline_news_clean.jsonl` | JSONL đã preprocess từ `src.data_ingestion`. |
| `--output-dir` | `src/chunking/output` | Thư mục ghi file chunk JSONL, summary và report. |
| `--strategies` | cả 4 strategy | Danh sách strategy cần chạy. Có thể truyền một hoặc nhiều giá trị: `token`, `langchain_recursive`, `llamaindex`, `structured`. |
| `--chunk-size` | `450` | Kích thước cửa sổ. Với `token`, `llamaindex` và `structured`, giá trị này gần với token count ước lượng. Với `langchain_recursive`, đây là character window theo thiết kế của LangChain/fallback. |
| `--overlap` | `80` | Độ chồng lặp giữa hai chunk liên tiếp. Với `token`, đây là số token ước lượng. Với `langchain_recursive`, đây là số ký tự. Với sentence-based fallback, đây là số token mục tiêu để giữ lại qua các câu cuối. |
| `--min-chunk-tokens` | `80` | Nếu chunk cuối ngắn hơn ngưỡng này, gộp chunk cuối vào chunk trước để tránh mảnh quá nhỏ. |
| `--small-article-chars` | `1000` | Bài có content không vượt quá số ký tự này được giữ thành một chunk duy nhất trước khi chạy strategy. |
| `--max-chunks-per-article` | không giới hạn | Giới hạn số chunk mỗi bài, hữu ích khi debug nhanh hoặc cần chặn bài quá dài. |
| `--no-title-injection` | tắt | Tắt việc chèn title/description/category vào trường `text`; `chunk_text` luôn là nội dung chunk gốc. |
| `--limit` | không giới hạn | Chỉ đọc N bài đầu tiên từ input để thử nghiệm nhanh. |
| `--no-progress` | tắt | Tắt progress bar `tqdm`. |

Lưu ý: `chunk_size` và `overlap` không có cùng đơn vị tuyệt đối ở mọi strategy. Khi so sánh kết quả, nên đọc thêm `avg_chunk_tokens`, `avg_chunk_chars`, `min_chunk_tokens` và `max_chunk_tokens` trong `chunking_summary.json`.

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
