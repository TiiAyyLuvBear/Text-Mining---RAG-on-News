# Embedding and Dense Retrieval

Module này dùng các chunk đã tạo trong `src/chunking/output` để xây dense index cho thí nghiệm retrieval. Toàn bộ output mặc định được ghi trong `src/embed/output`.

## Model mặc định

V1 dùng `intfloat/multilingual-e5-large` qua `sentence-transformers`.

- Phù hợp retrieval đa ngôn ngữ, trong đó có tiếng Việt.
- Vector dimension: `1024`.
- Nên normalize embeddings và dùng dot product, tương đương cosine similarity trên vector đã normalize.
- E5 cần prefix đúng kiểu huấn luyện:
  - Chunk/document: `passage: ...`
  - Query: `query: ...`

Chunk JSONL hiện tại đã có trường `text` chứa title, description, category và nội dung chunk. Vì vậy embedding dùng:

```text
passage: {row["text"]}
```

## Embed một strategy

Ví dụ với baseline `token`:

```powershell
python -m src.embed.embed_chunks `
  --input src/chunking/output/vieonline_news_chunks_token.jsonl `
  --output-dir src/embed/output/dense/token `
  --model intfloat/multilingual-e5-large `
  --batch-size 16 `
  --show-samples `
  --sample-size 5
```

CLI dùng `tqdm` để hiển thị tiến độ theo batch, tốc độ xử lý và ETA. Nếu muốn tắt progress bar:

```powershell
python -m src.embed.embed_chunks `
  --input src/chunking/output/vieonline_news_chunks_token.jsonl `
  --output-dir src/embed/output/dense/token `
  --no-progress
```

## Output

Mỗi strategy nên ghi vào một thư mục riêng:

```text
src/embed/output/dense/token/
  embeddings.npy
  metadata.jsonl
  manifest.json
  embedding_stats.json
  debug_samples.jsonl
```

- `embeddings.npy`: ma trận vector `float32`.
- `metadata.jsonl`: metadata cùng thứ tự với vector.
- `manifest.json`: model, dimension, batch size, input/output path, thời gian chạy.
- `embedding_stats.json`: thống kê số chunk, tốc độ embed, phân bố độ dài, strategy/category/implementation.
- `debug_samples.jsonl`: vài preview để kiểm tra text đưa vào model.

## Search thử

Sau khi embed:

```powershell
python -m src.embed.dense_search `
  --index-dir src/embed/output/dense/token `
  --query "Tin tức về công nghệ AI tại Việt Nam" `
  --top-k 5
```

Search encode query theo:

```text
query: {question}
```

Kết quả trả về `chunk_id`, `article_id`, `score`, `title`, `category`, `strategy`, `chunk_index`, `text` và `chunk_text`.

## So sánh input embedding giữa các strategy

Để xem cùng một bài được chunk khác nhau thế nào trước khi embed:

```powershell
python -m src.embed.compare_embedding_inputs `
  --inputs src/chunking/output/vieonline_news_chunks_token.jsonl `
           src/chunking/output/vieonline_news_chunks_langchain_recursive.jsonl `
           src/chunking/output/vieonline_news_chunks_llamaindex.jsonl `
           src/chunking/output/vieonline_news_chunks_structured.jsonl `
  --sample-articles 3 `
  --output src/embed/output/embedding_strategy_samples.md
```

Report cho thấy số chunk mỗi strategy, implementation, structure, token count và preview text được embed.

## Gợi ý debug

- Nếu `embedding_stats.json` có `token_stats.max` quá cao, kiểm tra `longest_chunks`; E5 sẽ truncate input dài.
- `langchain_recursive` đang dùng `chunk_size` theo character window nên thường sinh nhiều chunk ngắn hơn.
- `token` và `structured` thường gần nhau về độ dài trung bình, nhưng `structured` giữ thêm `structure=paragraph_xxx`.
- Khi so sánh retrieval, embed từng strategy vào thư mục riêng rồi chạy cùng một tập query/evaluator.




## Notebook tự động hóa pipeline

Notebook `src/embed/embedding_pipeline.ipynb` chạy tuần tự bằng các CLI hiện có:

1. `src.data_ingestion.cli` để tạo JSONL sạch.
2. `src.chunking.cli` để tạo chunk theo strategy.
3. `src.embed.compare_embedding_inputs` để xem mẫu khác biệt giữa strategy.
4. `src.embed.embed_chunks` để embed từng strategy với `tqdm` progress.
5. Đọc `embedding_stats.json` và lập bảng tổng hợp tại `src/embed/output/embedding_stats_summary.csv`.

Khi chạy CPU, giữ `ARTICLE_LIMIT` nhỏ và `BATCH_SIZE` thấp trong cell cấu hình trước khi chạy full corpus.

## Eval nhanh trên QA mẫu

Notebook có cell eval retrieval trên khoảng 50 QA mẫu từ `Dataset/QA_Claude/QA_output.csv`. Qrels được lấy theo `source_article_ids` hoặc `article_id`, và retrieved chunk được tính đúng nếu metadata `article_id` trùng qrels.

Output eval:

```text
src/embed/output/eval/leaderboard_qa50.csv
src/embed/output/eval/per_query_results_qa50.jsonl
```

Metric xếp hạng chính là `nDCG@10`; các metric phụ gồm `Recall@5`, `Recall@10`, `MRR@10`, `Hit@1`, `Hit@5`, latency truy vấn, index size, embedding dimension, số chunk và avg chunk tokens.
