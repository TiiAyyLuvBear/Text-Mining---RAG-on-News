# Data Ingestion

Module này tiền xử lí dữ liệu VieOnlineNews từ CSV thô sang JSONL sạch để dùng cho các bước chunking, embedding và retrieval.

## Input

CSV đầu vào phải có đủ các cột:

| Cột | Ý nghĩa |
| --- | --- |
| `id` | Mã bài viết gốc. |
| `title` | Tiêu đề bài báo. |
| `description` | Mô tả ngắn hoặc sapo. |
| `content` | Nội dung chính của bài báo. |
| `category` | Chuyên mục bài viết. |

Nếu một dòng thiếu bất kỳ cột bắt buộc nào hoặc giá trị rỗng ở các cột này, dòng đó không được ghi vào output và được tính vào `missing_required`.

## Pipeline Tiền Xử Lí

Với mỗi bài viết hợp lệ, pipeline thực hiện các bước sau:

1. Chuẩn hóa Unicode về NFC để giảm lỗi khác biệt dấu tiếng Việt.
2. Giải mã HTML entity bằng `html.unescape`, ví dụ `&amp;`.
3. Loại HTML tag bằng regex `<[^>]+>`.
4. Tùy chọn loại URL nếu bật `--strip-urls`; mặc định chỉ đánh dấu URL trong `quality_flags`.
5. Sửa một số lỗi crawl phổ biến do mất khoảng trắng sau hyperlink, ví dụ `Theo...`, `Ảnh...`, `Video...` và một nhóm từ chức năng tiếng Việt bị dính.
6. Chuẩn hóa khoảng trắng, giữ ngắt đoạn bằng `\n`, loại dòng rỗng và khoảng trắng dư.
7. Tạo trường `text` cho embedding từ title, description, category và content; hoặc chỉ dùng content nếu bật `--content-only`.
8. Gắn `quality_flags` và metadata phục vụ kiểm tra chất lượng.

Pipeline không tự dịch, tóm tắt, suy diễn nội dung hoặc loại bỏ bài viết chỉ vì có quality flag. Các flag dùng để lọc hoặc review ở bước sau.

## Chạy CLI

Chạy bằng module:

```powershell
python -m src.data_ingestion.cli `
  --input Dataset/VieOnlineNews.csv `
  --output data/processed/vieonline_news_clean.jsonl
```

Hoặc chạy trực tiếp file CLI:

```powershell
python src\data_ingestion\cli.py `
  --input Dataset\VieOnlineNews.csv `
  --output data\processed\vieonline_news_clean.jsonl
```

Mặc định CLI cũng ghi file review:

```text
data/processed/vieonline_news_human_review.jsonl
```

File review chỉ chứa một số dòng cần xem lại: bài có `quality_flags` hoặc nội dung sau khi clean khác nội dung thô.

## Tham Số

| Tham số | Mặc định | Ý nghĩa |
| --- | ---: | --- |
| `--input` | `Dataset/VieOnlineNews.csv` | Đường dẫn CSV thô. |
| `--output` | `data/processed/vieonline_news_clean.jsonl` | Đường dẫn JSONL sạch. Thư mục cha được tạo tự động. |
| `--review-output` | `data/processed/vieonline_news_human_review.jsonl` | JSONL phục vụ human review. Truyền chuỗi rỗng `""` để tắt. |
| `--review-limit` | `200` | Số dòng review tối đa được ghi. |
| `--min-content-chars` | `300` | Nếu content sau clean ngắn hơn ngưỡng này, gắn flag `short_content`. |
| `--long-content-chars` | `20000` | Nếu content sau clean dài hơn ngưỡng này, gắn flag `long_content`. |
| `--strip-urls` | tắt | Xóa URL khỏi text. Nếu không bật, URL được giữ lại nhưng bài có URL sẽ nhận flag `url_like`. |
| `--content-only` | tắt | Trường `text` chỉ chứa content sạch, không chèn title/description/category. |

## Output Schema

Mỗi dòng trong JSONL output là một article record:

```json
{
  "article_id": "123",
  "title": "Tiêu đề sạch",
  "description": "Mô tả sạch",
  "content": "Nội dung sạch",
  "category": "Chuyên mục",
  "text": "Tiêu đề: ...\nMô tả: ...\nChuyên mục: ...\nNội dung:\n...",
  "metadata": {
    "article_id": "123",
    "title": "Tiêu đề sạch",
    "description": "Mô tả sạch",
    "category": "Chuyên mục",
    "content_chars": 1234,
    "title_chars": 42,
    "description_chars": 120,
    "is_short": false,
    "is_long": false
  },
  "quality_flags": []
}
```

Các trường quan trọng:

- `text`: nội dung sẽ được dùng ở các bước embedding hoặc chunking nếu cần ngữ cảnh đầy đủ.
- `content`: nội dung bài báo sạch, được module chunking dùng làm nguồn cắt chunk.
- `metadata`: thông tin lặp lại để downstream không phải parse lại article-level fields.
- `quality_flags`: danh sách cảnh báo chất lượng.

## Quality Flags Và Stats

Các flag hiện có:

| Flag | Điều kiện |
| --- | --- |
| `short_content` | `len(content) < min_content_chars`. |
| `long_content` | `len(content) > long_content_chars`. |
| `html_like` | Nội dung thô có HTML tag hoặc HTML entity dạng `&name;`. |
| `url_like` | Nội dung thô có URL dạng `http(s)://...` hoặc `www...`. |

Sau khi chạy, CLI in JSON stats:

| Chỉ số | Ý nghĩa |
| --- | --- |
| `rows_read` | Số dòng CSV đã đọc. |
| `rows_written` | Số article record đã ghi. |
| `missing_required` | Số dòng bị bỏ qua vì thiếu trường bắt buộc. |
| `short_content` | Số record có flag `short_content`. |
| `long_content` | Số record có flag `long_content`. |
| `html_like` | Số record có flag `html_like`. |
| `url_like` | Số record có flag `url_like`. |

## Gợi Ý Cấu Hình

- Giữ mặc định `--content-only` tắt khi muốn embedding có thêm tiêu đề, mô tả và chuyên mục để tăng ngữ cảnh semantic.
- Bật `--content-only` nếu bước sau đã tự inject metadata, hoặc muốn đánh giá riêng chất lượng nội dung thuần.
- Không bật `--strip-urls` trong lần chạy đầu để review được mức độ nhiễu URL; bật sau nếu URL làm embedding lệch chủ đề.
- Tăng `--review-limit` khi cần audit dữ liệu rộng hơn trước khi chunking toàn bộ corpus.
