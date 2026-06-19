# Vietnamese News QA System using Text Mining and Retrieval-Augmented Generation

## 1. Tên đề tài và mục tiêu

**Tên đề tài:** Vietnamese News QA System using Text Mining and Retrieval-Augmented Generation.

**Mục tiêu:** xây dựng hệ thống hỏi đáp trên dữ liệu báo chí tiếng Việt, tập trung nghiên cứu hiệu quả của các phương pháp Information Retrieval trong kiến trúc RAG.

Đồ án cần trả lời 5 câu hỏi chính:

1. BM25, Dense Retrieval hay Hybrid Retrieval phù hợp nhất cho dữ liệu tin tức tiếng Việt?
2. Reranking có cải thiện chất lượng retrieval không?
3. Retrieval tốt hơn có làm chất lượng trả lời QA tốt hơn không?
4. Hệ thống có biết từ chối khi câu hỏi không có bằng chứng trong corpus không?
5. Lỗi chính nằm ở dữ liệu, retrieval, chunking hay generation?

## 2. Input và Output hệ thống

### Input

Người dùng nhập câu hỏi tiếng Việt tự nhiên, ví dụ:

```text
Các chính sách kinh tế mới nhất được nhắc đến trong tuần qua là gì?
```

### Pipeline xử lý

1. Nhận câu hỏi từ người dùng.
2. Chuẩn hóa câu hỏi: Unicode, khoảng trắng, lowercase nếu cần.
3. Embed câu hỏi nếu dùng Dense/Hybrid.
4. Truy xuất top-k đoạn tin bằng BM25, Dense hoặc Hybrid.
5. Rerank top candidates nếu dùng reranker.
6. Chọn 3-5 passage tốt nhất làm context.
7. Đưa context vào LLM để sinh câu trả lời.
8. Trả về câu trả lời kèm nguồn: title, URL nếu có, hoặc article_id.

### Output

Output cần có:

- Câu trả lời tiếng Việt tự nhiên, đúng trọng tâm.
- Danh sách nguồn để người dùng kiểm chứng.
- Nếu không đủ bằng chứng, hệ thống phải nói rõ không tìm thấy đủ thông tin.

Ví dụ:

```text
Theo các bài báo được truy xuất, chính sách kinh tế được nhắc đến tập trung vào hỗ trợ doanh nghiệp, kiểm soát lạm phát và thúc đẩy đầu tư công. Tuy nhiên, dữ liệu hiện tại không đủ để kết luận đây là toàn bộ chính sách trong tuần qua.

Nguồn:
1. article_id=12345 - Tên bài báo A
2. article_id=67890 - Tên bài báo B
```

## 3. Công nghệ và mô hình dự kiến

### Retrieval

- **BM25:** `rank_bm25` cho baseline nhanh; Elasticsearch/OpenSearch nếu muốn bản search engine chuyên nghiệp hơn.
- **Dense Retrieval:** `BAAI/bge-m3` hoặc Cohere Embed v3 Multilingual.
- **Hybrid Retrieval:** kết hợp BM25 + Dense bằng score fusion hoặc Reciprocal Rank Fusion.

### Embedding Model

- **BAAI/bge-m3:** open-source, multilingual, phù hợp tiếng Việt, có thể chạy local nếu đủ tài nguyên.
- **Cohere Embed v3 Multilingual:** API ổn định, embedding mạnh, phù hợp nếu muốn kết quả nhanh và có API key.

### Re-ranking Model

- **BAAI/bge-reranker-m3:** reranker open-source.
- **Cohere Rerank v3:** reranker API.

### LLM sinh câu trả lời

- **Qwen2.5-7B-Instruct:** hướng open-source/local.
- **GPT-4o:** API mạnh, dùng làm baseline chất lượng cao.
- **Claude Opus 4.6:** API chất lượng cao nếu có quyền truy cập.
- **Gemini 5.5:** API chất lượng cao nếu có quyền truy cập.

### Vector Database

- **Qdrant Cloud/local:** khuyến nghị cho đồ án vì dễ dùng và dễ debug.
- **Pinecone Enterprise Plan:** phù hợp nếu nhóm có tài khoản enterprise/credit.

## 4. Hiện trạng repo

Repo hiện đã có các file nền tảng:

```text
Dataset/
├── Create_QA_Vietonline/
│   ├── CONVERT_QA.py
│   ├── data.py
│   ├── QA.py
│   └── README.md
├── QA_Claude/
│   ├── QA_output.csv
│   └── QA_output.jsonl

src/
├── eda_corpus.py
├── validate_qa.py
├── chunk_corpus.py
├── _utf8_head.py
└── README.md
```

Ý nghĩa các file đã có:

- `src/eda_corpus.py`: EDA corpus, kiểm tra null/trùng/lệch dữ liệu, convert CSV sang Parquet.
- `src/validate_qa.py`: kiểm tra QA set có khớp corpus không.
- `src/chunk_corpus.py`: cắt bài báo thành passage/chunk.
- `src/README.md`: hướng dẫn chạy các script chuẩn bị dữ liệu.

Vì vậy phần tiếp theo không bắt đầu từ số 0, mà cần tiếp tục theo hướng: chạy report thật, kiểm QA, xây retriever, evaluate, ghép RAG, viết báo cáo.

## 5. Cấu trúc project nên hoàn thiện

Nên chuẩn hóa repo thành:

```text
Text-Mining---RAG-on-News/
├── Dataset/
│   ├── VietOnlineNews_CSV/
│   ├── parquet/
│   ├── chunks/
│   ├── indexes/
│   └── QA_Claude/
├── configs/
│   ├── bm25.yaml
│   ├── dense_bge_m3.yaml
│   ├── dense_cohere.yaml
│   ├── hybrid.yaml
│   └── rag.yaml
├── reports/
│   ├── eda/
│   ├── qa_validation/
│   ├── retrieval_eval/
│   └── rag_eval/
├── src/
│   ├── eda_corpus.py
│   ├── validate_qa.py
│   ├── chunk_corpus.py
│   ├── retrieval/
│   ├── evaluation/
│   └── rag/
├── requirements.txt
├── .env.example
+└── README.md
```

## 6. Phase 0 — Chuẩn hóa môi trường và repo

### Mục tiêu

Đảm bảo người khác clone repo có thể hiểu và chạy lại toàn bộ pipeline.

### Việc cần làm

- Tạo `requirements.txt`.
- Tạo `.env.example` cho API key: Cohere, OpenAI, Pinecone, Qdrant.
- Tạo `reports/` để lưu kết quả EDA/evaluation.
- Tạo `configs/` để lưu cấu hình experiment.
- Tạo `src/retrieval/` cho BM25, Dense, Hybrid.
- Tạo `src/evaluation/` cho metric retrieval và QA.
- Tạo `src/rag/` cho prompt, generator, citation.

### Output cần có

```text
requirements.txt
.env.example
configs/*.yaml
reports/
src/retrieval/
src/evaluation/
src/rag/
```

### Checklist hoàn thành

- Có hướng dẫn cài dependencies.
- Có hướng dẫn chạy từng phase.
- Không hard-code API key trong code.
- File sinh ra được lưu đúng thư mục.

## 7. Phase 1 — EDA và chuẩn hóa corpus

### Mục tiêu

Hiểu rõ corpus trước khi xây retrieval.

Cần thống kê:

- Số bài báo từng split.
- Số category.
- Phân bố category.
- Độ dài title/description/content.
- Số null theo cột.
- Số bài trùng ID.
- Số bài content rỗng.
- Số bài content quá ngắn.
- Encoding tiếng Việt có lỗi không.

### Input

```text
Dataset/VietOnlineNews_CSV/train.csv
Dataset/VietOnlineNews_CSV/validation.csv
Dataset/VietOnlineNews_CSV/test.csv
```

Các cột nên có:

- `id`
- `title`
- `description`
- `content`
- `category`
- `url` nếu có

### Việc cần làm

- Chạy `src/eda_corpus.py`.
- Convert CSV sang Parquet.
- Lưu log EDA vào `reports/eda/corpus_summary.md`.
- Xuất bảng category distribution.
- Xuất bảng content length stats.
- Kiểm tra duplicate ID trong từng split.
- Kiểm tra duplicate ID giữa train/validation/test.
- Kiểm tra bài có content rỗng hoặc content rất ngắn.
- Vẽ biểu đồ phân bố độ dài content nếu có thời gian.

### Command

```powershell
python .\src\eda_corpus.py --data-dir Dataset\VietOnlineNews_CSV --out-dir Dataset\parquet
```

### Output

```text
Dataset/parquet/train.parquet
Dataset/parquet/validation.parquet
Dataset/parquet/test.parquet
reports/eda/corpus_summary.md
reports/eda/category_distribution.csv
reports/eda/content_length_stats.csv
```

### Quyết định cần chốt

- Có xóa bài content rỗng không?
- Có xóa bài content quá ngắn không?
- Nếu duplicate ID thì giữ bản nào?
- Nếu thiếu URL, citation dùng `article_id` + `title`.

### Con người cần check QA/data

- Mở ngẫu nhiên 30 bài báo.
- Kiểm tra title/content đúng tiếng Việt.
- Kiểm tra category hợp lý.
- Kiểm tra content không bị lỗi font.
- Kiểm tra bài không bị cắt cụt bất thường.

## 8. Phase 2 — Validate QA set

### Mục tiêu

Đảm bảo QA set đủ tin cậy để làm ground truth đánh giá.

Nếu QA sai, mọi metric retrieval và RAG đều sai theo.

### Input

```text
Dataset/QA_Claude/QA_output.jsonl
Dataset/QA_Claude/QA_output.csv
Dataset/parquet/*.parquet
```

QA record nên có:

- `id` hoặc `qa_id`
- `question`
- `answers`
- `article_id`
- `is_possible`
- `qa_type`

### Việc cần làm bằng script

- Chạy `src/validate_qa.py`.
- Đếm tổng số QA.
- Đếm `is_possible=true`.
- Đếm `is_possible=false`.
- Đếm single-article QA.
- Đếm cross-article QA.
- Kiểm tra `article_id` có trong corpus không.
- Kiểm tra `is_possible=true` nhưng answer rỗng.
- Kiểm tra `is_possible=false` nhưng vẫn có answer.
- Kiểm tra câu hỏi trùng nhau.
- Kiểm tra bài được tham chiếu nằm ở split nào.

### Command

```powershell
python .\src\validate_qa.py
```

Nếu sau này refactor script, nên hỗ trợ:

```powershell
python .\src\validate_qa.py --qa Dataset\QA_Claude\QA_output.jsonl --parquet-dir Dataset\parquet --out reports\qa_validation
```

### Output

```text
reports/qa_validation/qa_summary.md
reports/qa_validation/missing_article_ids.csv
reports/qa_validation/problematic_qa.csv
reports/qa_validation/qa_type_distribution.csv
```

### Bước con người check toàn bộ QA

Đây là bước bắt buộc, không được bỏ.

Checklist human QA:

- Mở từng QA hoặc sample lớn nếu QA quá nhiều.
- Với `is_possible=true`, mở bài gốc theo `article_id`.
- Xác nhận câu hỏi trả lời được từ bài báo.
- Xác nhận answer đúng, không bịa, không thêm thông tin ngoài bài.
- Xác nhận câu hỏi không quá mơ hồ.
- Với `is_possible=false`, xác nhận thật sự không đủ bằng chứng.
- Với cross-article QA, xác nhận cần nhiều bài để trả lời.
- Đánh dấu lỗi theo nhãn: `wrong_answer`, `missing_evidence`, `ambiguous_question`, `bad_article_id`, `duplicate_question`, `format_error`.
- Tạo file cuối cùng: `Dataset/QA_Claude/QA_reviewed.jsonl`.

### Quy tắc xử lý QA lỗi

- QA lỗi nhẹ: sửa question/answer.
- QA sai article_id: sửa article_id nếu tìm được bài đúng.
- QA mơ hồ: loại khỏi metric chính hoặc đưa vào nhóm phân tích riêng.
- QA không sửa được: loại khỏi evaluation set.
- Chỉ dùng `QA_reviewed.jsonl` cho kết quả cuối kỳ.

## 9. Phase 3 — Chunking và tiền xử lý passage

### Mục tiêu

Chuyển bài báo dài thành passage phù hợp cho retrieval/RAG.

Đơn vị retrieval chính là chunk, không phải toàn bộ bài báo.

### Input

```text
Dataset/parquet/train.parquet
Dataset/parquet/validation.parquet
Dataset/parquet/test.parquet
```

### Việc cần làm

- Chạy `src/chunk_corpus.py`.
- Thử nhiều cấu hình chunk:
  - 256 tokens, overlap 50.
  - 384 tokens, overlap 64.
  - 512 tokens, overlap 100.
- Mỗi chunk cần metadata:
  - `chunk_id`
  - `article_id`
  - `title`
  - `description`
  - `category`
  - `url`
  - `split`
  - `chunk_index`
  - `n_tokens`
  - `text`
- Kiểm tra chunk không rỗng.
- Kiểm tra mỗi article có content thì có ít nhất 1 chunk.
- Thống kê số chunk/bài.
- Thống kê độ dài chunk.

### Command

```powershell
python .\src\chunk_corpus.py --parquet-dir Dataset\parquet --out Dataset\chunks --max-tokens 384 --overlap 64
```

### Output

```text
Dataset/chunks/train_chunks.jsonl
Dataset/chunks/validation_chunks.jsonl
Dataset/chunks/test_chunks.jsonl
reports/eda/chunk_stats.md
```

### Quyết định cần chốt

- Dùng chunk size nào cho experiment chính?
- Index toàn bộ corpus hay chỉ train?
- QA tham chiếu split nào thì index có bao gồm split đó không?

Khuyến nghị: nếu QA được tạo từ toàn bộ corpus, index toàn bộ corpus và ghi rõ điều này trong báo cáo.

### Con người cần check

- Mở ngẫu nhiên 30 chunk.
- Kiểm tra chunk có đủ ngữ cảnh.
- Kiểm tra overlap không quá lặp.
- Kiểm tra title/category/article_id đúng.
- Kiểm tra tiếng Việt không lỗi dấu.

## 10. Phase 4 — Xây BM25 Retriever

### Mục tiêu

Tạo baseline retrieval bằng keyword để so sánh với Dense và Hybrid.

BM25 quan trọng vì dễ giải thích, không cần GPU/API, và thường mạnh khi câu hỏi trùng từ khóa với bài báo.

### Việc cần làm

- Tạo `src/retrieval/bm25_index.py`.
- Tạo `src/retrieval/bm25_search.py`.
- Đọc chunks từ `Dataset/chunks/*_chunks.jsonl`.
- Tiền xử lý text:
  - Unicode normalize.
  - Lowercase.
  - Xóa khoảng trắng thừa.
  - Tokenize bằng whitespace trước.
  - Thử `underthesea` hoặc `pyvi` nếu cần tách từ tiếng Việt.
- Build BM25 bằng `rank_bm25`.
- Lưu index vào `Dataset/indexes/bm25.pkl`.
- Search trả về `chunk_id`, `article_id`, `score`, `title`, `text`.

### Output

```text
Dataset/indexes/bm25.pkl
reports/retrieval_eval/bm25_sample_results.jsonl
```

### Test nhanh

- Chạy 10 câu hỏi mẫu.
- Xem top-5 có liên quan không.
- Đo latency trung bình.

### Con người cần check

- Với 20 câu QA, xem top-5 BM25 có chứa bài đúng không.
- Nếu fail, ghi nguyên nhân: khác từ khóa, tokenization kém, chunk thiếu context, QA sai.

## 11. Phase 5 — Xây Dense Retriever

### Mục tiêu

Tạo retriever ngữ nghĩa để tìm đoạn liên quan ngay cả khi câu hỏi không trùng keyword.

### Model cần thử

- `BAAI/bge-m3` nếu muốn local/open-source.
- Cohere Embed v3 Multilingual nếu có API key.

### Việc cần làm

- Tạo `src/retrieval/embed_chunks.py`.
- Tạo `src/retrieval/dense_index.py`.
- Tạo `src/retrieval/dense_search.py`.
- Chuẩn hóa input embedding theo format:

```text
Title: {title}
Category: {category}
Content: {text}
```

- Embed toàn bộ chunks theo batch.
- Cache embedding local hoặc upsert vào vector DB.
- Nếu dùng Qdrant: tạo collection, cosine distance, upsert vector + payload.
- Nếu dùng Pinecone: tạo index đúng dimension, upsert vector + metadata.
- Search question và trả về top-k chunks.

### Output

Nếu lưu local:

```text
Dataset/indexes/dense_embeddings.npy
Dataset/indexes/dense_metadata.jsonl
```

Nếu dùng vector DB:

```text
reports/retrieval_eval/vector_db_collection_info.md
```

### Log cần lưu

- Model embedding.
- Dimension.
- Số chunk embed.
- Batch size.
- Thời gian embed.
- Thời gian build index.
- Search latency.
- Chi phí API nếu có.

### Con người cần check

- Với 20 câu hỏi diễn đạt khác keyword, xem Dense có tìm đúng hơn BM25 không.
- Kiểm tra false positive: đoạn nghe liên quan nhưng không chứa bằng chứng trả lời.

## 12. Phase 6 — Xây Hybrid Retriever và Reranker

### Mục tiêu

Kết hợp ưu điểm BM25 và Dense.

### Hybrid việc cần làm

- Tạo `src/retrieval/hybrid_search.py`.
- Lấy top 50 từ BM25.
- Lấy top 50 từ Dense.
- Gộp candidates theo `chunk_id`.
- Normalize score bằng min-max, z-score hoặc Reciprocal Rank Fusion.
- Thử công thức:

```text
hybrid_score = alpha * dense_score + (1 - alpha) * bm25_score
```

- Thử alpha: 0.2, 0.5, 0.8.
- Lưu result từng alpha.

### Reranking việc cần làm

- Lấy top 50 hybrid candidates.
- Rerank bằng `BAAI/bge-reranker-m3` hoặc Cohere Rerank v3.
- Chọn top 5 hoặc top 10 cuối cùng.
- Đo rerank có cải thiện MRR/nDCG không.

### Output

```text
reports/retrieval_eval/hybrid_alpha_0.2.jsonl
reports/retrieval_eval/hybrid_alpha_0.5.jsonl
reports/retrieval_eval/hybrid_alpha_0.8.jsonl
reports/retrieval_eval/hybrid_rerank.jsonl
```

### Con người cần check

- So sánh top-5 BM25, Dense, Hybrid trên cùng 20 câu.
- Ghi câu nào Hybrid tốt hơn.
- Ghi câu nào Hybrid kém hơn.
- Kiểm tra reranker có đẩy evidence đúng lên cao không.

## 13. Phase 7 — Đánh giá Retrieval

### Mục tiêu

Đánh giá định lượng chất lượng truy xuất trên QA set. Đây là phần trọng tâm của đồ án.

### Ground truth

- Single-article QA đúng nếu top-k có `article_id` trùng ground truth.
- Cross-article QA đúng nếu top-k có một hoặc nhiều `article_id` liên quan.
- Nếu có evidence chunk, đánh giá chính xác ở chunk-level.
- Nếu chỉ có article_id, đánh giá ở article-level.

### Metric bắt buộc

- Recall@1
- Recall@3
- Recall@5
- Recall@10
- MRR@10
- nDCG@10

### Metric nên thêm

- Hit Rate@k.
- Mean search latency.
- Index size.
- API cost nếu có.

### Việc cần làm

- Tạo `src/evaluation/evaluate_retrieval.py`.
- Input: `QA_reviewed.jsonl` và retrieval results.
- Chạy evaluation cho:
  - BM25.
  - Dense bge-m3.
  - Dense Cohere nếu có.
  - Hybrid alpha 0.2.
  - Hybrid alpha 0.5.
  - Hybrid alpha 0.8.
  - Hybrid + rerank.
- Tách metric theo QA type: single, cross, impossible.
- Tách metric theo category nếu cần.

### Output

```text
reports/retrieval_eval/retrieval_metrics.csv
reports/retrieval_eval/retrieval_metrics.md
reports/retrieval_eval/error_analysis.jsonl
```

### Bảng báo cáo cần có

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Dense bge-m3 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Dense Cohere | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Hybrid | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Hybrid + Rerank | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Con người cần check

- Lấy toàn bộ câu fail ở Recall@10.
- Phân loại lỗi: QA sai, chunking mất evidence, retriever fail, bài không nằm trong index, ground truth thiếu.
- Tạo `reports/retrieval_eval/error_analysis_notes.md`.

## 14. Phase 8 — Ghép RAG pipeline

### Mục tiêu

Dùng retriever tốt nhất làm context provider cho LLM.

### Việc cần làm

- Tạo `src/rag/prompt_templates.py`.
- Tạo `src/rag/generate_answer.py`.
- Tạo `src/rag/rag_pipeline.py`.
- Thiết kế prompt bắt buộc LLM chỉ dùng context.
- Nếu context không đủ, LLM phải từ chối.
- Lấy top 3-5 chunks.
- Gộp context theo format có source rõ ràng.
- Gọi LLM.
- Trả câu trả lời + citations.

### Prompt mẫu

```text
Bạn là hệ thống hỏi đáp tin tức tiếng Việt.
Chỉ sử dụng thông tin trong CONTEXT để trả lời QUESTION.
Nếu CONTEXT không chứa đủ bằng chứng, hãy trả lời: "Không tìm thấy đủ thông tin trong dữ liệu được cung cấp."
Luôn kèm nguồn ở cuối câu trả lời.

QUESTION:
question}

CONTEXT:
[1] article_id={article_id}, title={title}, url={url}
{text}

OUTPUT:
- Câu trả lời:
- Nguồn:
```

### Biến thể cần thử

- RAG với BM25 top-5.
- RAG với Dense top-5.
- RAG với Hybrid top-5.
- RAG với Hybrid + rerank top-5.
- Thử top-3, top-5, top-10 nếu có thời gian.

### Output

```text
reports/rag_eval/rag_predictions_bm25.jsonl
reports/rag_eval/rag_predictions_dense.jsonl
reports/rag_eval/rag_predictions_hybrid.jsonl
reports/rag_eval/rag_predictions_hybrid_rerank.jsonl
```

### Con người cần check

- Đọc sample 50 câu trả lời.
- Kiểm tra answer có hallucination không.
- Kiểm tra citation có chứa bằng chứng không.
- Kiểm tra câu unanswerable có bị LLM bịa không.
- Gắn lỗi: `retrieval_error`, `generation_error`, `hallucination`, `missing_citation`, `wrong_refusal`, `should_refuse_but_answered`.

## 15. Phase 9 — Đánh giá QA/RAG

### Mục tiêu

Đo chất lượng câu trả lời cuối cùng và chứng minh ảnh hưởng của retrieval lên QA.

### Metric cho answerable QA

- Exact Match nếu answer ngắn.
- Token-level F1 cho tiếng Việt.
- ROUGE-L nếu answer dài/tóm tắt.
- LLM-as-a-judge nếu cần semantic evaluation, nhưng phải ghi rõ hạn chế.

### Metric cho unanswerable QA

- Accuracy từ chối trả lời.
- False positive rate.
- Refusal quality.

### Metric citation

- Citation exists rate.
- Citation correctness sample-based.
- Answer supported by cited context.

### Việc cần làm

- Tạo `src/evaluation/evaluate_qa.py`.
- Chuẩn hóa text tiếng Việt: lowercase, Unicode normalize, bỏ khoảng trắng thừa, cân nhắc bỏ dấu câu.
- Tính EM/F1/ROUGE-L.
- Tính unanswerable accuracy.
- Tính citation rate.
- Tách kết quả theo retriever và QA type.

### Output

```text
reports/rag_eval/qa_metrics.csv
reports/rag_eval/qa_metrics.md
reports/rag_eval/qa_error_analysis.jsonl
```

### Bảng cần có

| RAG Context Source | EM | F1 | ROUGE-L | Unanswerable Acc | Citation Rate |
|---|---:|---:|---:|---:|---:|
| BM25 | TBD | TBD | TBD | TBD | TBD |
| Dense | TBD | TBD | TBD | TBD | TBD |
| Hybrid | TBD | TBD | TBD | TBD | TBD |
| Hybrid + Rerank | TBD | TBD | TBD | TBD | TBD |

### Con người cần check

- Nếu retrieval có bài đúng nhưng LLM trả lời sai: lỗi generation.
- Nếu retrieval không có bài đúng: lỗi retrieval.
- Nếu ground truth sai: lỗi dataset.
- Tạo bảng ví dụ lỗi tiêu biểu.

## 16. Phase 10 — Error analysis chuyên sâu

### Mục tiêu

Giải thích vì sao hệ thống sai và rút ra kết luận nghiên cứu.

### Nhóm lỗi Retrieval

- Keyword mismatch.
- Tokenizer tiếng Việt kém.
- Dense nhầm semantic gần nhưng sai sự kiện.
- Chunk quá ngắn thiếu context.
- Chunk quá dài làm nhiễu embedding.
- Cross-article QA khó.
- Ground truth thiếu bài liên quan.

### Nhóm lỗi Generation

- LLM bịa thêm thông tin.
- LLM không đọc hết context.
- LLM trả lời quá chung.
- LLM citation sai.
- LLM không từ chối khi thiếu context.
- LLM từ chối dù context đủ.

### Output

```text
reports/final_error_analysis.md
```

Nội dung nên có:

- 5 ví dụ BM25 tốt hơn Dense.
- 5 ví dụ Dense tốt hơn BM25.
- 5 ví dụ Hybrid tốt nhất.
- 5 ví dụ reranker cải thiện thứ hạng.
- 5 ví dụ hallucination.
- 5 ví dụ unanswerable xử lý đúng.

## 17. Phase 11 — Demo ứng dụng

### Mục tiêu

Tạo demo để người dùng nhập câu hỏi và nhận câu trả lời có nguồn.

### Giao diện đề xuất

- Streamlit nếu cần UI nhanh.
- CLI nếu chỉ cần demo kỹ thuật.

### Việc cần làm

- Tạo `app.py` hoặc `src/app_streamlit.py`.
- Cho phép nhập question.
- Cho phép chọn retriever.
- Hiển thị top retrieved chunks.
- Hiển thị answer RAG.
- Hiển thị citation và retrieval score.

### Command demo

```powershell
streamlit run app.py
```

### Con người cần check

- Nhập 20 câu hỏi tự do.
- App không crash.
- Câu trả lời có nguồn.
- Câu ngoài phạm vi được từ chối hợp lý.
- Latency chấp nhận được.

## 18. Phase 12 — Viết báo cáo cuối kỳ

### Cấu trúc báo cáo đề xuất

1. Giới thiệu.
2. Mục tiêu nghiên cứu.
3. Dữ liệu và QA set.
4. EDA corpus.
5. Phương pháp chunking.
6. BM25 Retrieval.
7. Dense Retrieval.
8. Hybrid Retrieval và reranking.
9. RAG pipeline.
10. Thiết lập thí nghiệm.
11. Kết quả Retrieval.
12. Kết quả QA/RAG.
13. Error analysis.
14. Kết luận và hướng phát triển.

### Hình/bảng cần có

- Phân bố category.
- Phân bố độ dài content.
- Phân bố độ dài chunk.
- Bảng thống kê QA set.
- Bảng retrieval metrics.
- Bảng QA metrics.
- Sơ đồ pipeline RAG.
- Bảng ví dụ lỗi.

## 19. Thứ tự ưu tiên làm từ bây giờ

1. Chạy EDA thật bằng `src/eda_corpus.py` và lưu report.
2. Chạy validate QA bằng `src/validate_qa.py`.
3. Cho con người review QA và tạo `QA_reviewed.jsonl`.
4. Chạy chunking với 2-3 cấu hình.
5. Xây BM25 baseline.
6. Viết evaluator retrieval.
7. Xây Dense Retrieval bằng bge-m3 hoặc Cohere Embed v3.
8. Xây Hybrid Retrieval và thử alpha.
9. Thêm reranker nếu còn thời gian.
10. Chạy retrieval evaluation đầy đủ.
11. Chọn retriever tốt nhất.
12. Ghép RAG pipeline.
13. Đánh giá QA/RAG.
14. Làm demo Streamlit hoặc CLI.
15. Viết báo cáo và slide.

## 20. Milestone đề xuất

### Milestone 1 — Data Ready

- Parquet corpus có đủ train/validation/test.
- Chunk corpus đã tạo.
- QA reviewed đã chốt.
- Có report EDA.
- Có report QA validation.

### Milestone 2 — Retrieval Ready

- BM25 chạy được.
- Dense chạy được.
- Hybrid chạy được.
- Rerank chạy được nếu dùng.
- Có bảng Recall@k, MRR, nDCG.
- Có error analysis retrieval.

### Milestone 3 — RAG Ready

- RAG pipeline chạy end-to-end.
- Prompt rõ ràng.
- Answer có citation.
- Có xử lý câu không đủ thông tin.
- Có QA metrics.
- Có error analysis generation.

### Milestone 4 — Final Submission

- Demo chạy được.
- README hoàn chỉnh.
- Báo cáo cuối kỳ.
- Slide thuyết trình.
- Bảng kết quả cuối cùng.

## 21. Rủi ro và cách xử lý

### Corpus thiếu URL

- Dùng `article_id` và `title` làm citation.
- Nếu có thể, bổ sung URL từ nguồn ban đầu.

### QA set có lỗi

- Không dùng QA thô cho kết quả cuối.
- Bắt buộc human review.
- Ghi rõ số QA bị sửa/loại.

### Dense embedding quá chậm

- Embed theo batch.
- Cache embedding.
- Chạy subset trước.
- Dùng API nếu local không đủ tài nguyên.

### API tốn chi phí

- Chạy full bằng model local nếu có thể.
- API chỉ dùng cho subset hoặc final comparison.
- Log số request và chi phí ước tính.

### LLM hallucination

- Prompt yêu cầu chỉ dùng context.
- Thêm rule từ chối nếu thiếu bằng chứng.
- Đánh giá citation correctness riêng.

## 22. Checklist hoàn thành cuối đồ án

### Data

- Corpus đã EDA.
- Corpus đã convert sang Parquet.
- Chunk files đã tạo.
- QA set đã validate bằng script.
- QA set đã được con người kiểm tra.
- QA final đã chốt.

### Retrieval

- BM25 chạy được.
- Dense chạy được.
- Hybrid chạy được.
- Rerank chạy được nếu dùng.
- Có Recall@k, MRR, nDCG.
- Có error analysis.

### RAG/QA

- RAG pipeline chạy được.
- Prompt có rule chống hallucination.
- Answer có citation.
- Có xử lý unanswerable.
- Có EM/F1 hoặc metric tương đương.
- Có unanswerable accuracy.

### Demo và báo cáo

- Demo nhập câu hỏi được.
- App trả lời có nguồn.
- Báo cáo có dữ liệu, phương pháp, kết quả, phân tích lỗi, kết luận.
- Slide có pipeline, bảng metric và demo scenario.

## 23. Kết luận định hướng

Đồ án nên được trình bày như một nghiên cứu thực nghiệm về retrieval cho tiếng Việt, không chỉ là chatbot demo. Kết quả cuối cùng cần chứng minh bằng số liệu rằng phương pháp retrieval nào tốt nhất, reranking có đáng dùng không, và retrieval ảnh hưởng trực tiếp như thế nào đến chất lượng câu trả lời trong RAG.
