# HƯỚNG DẪN TẠO EMBEDDING VÀ DENSE RETRIEVAL (EMBEDDING & DENSE RETRIEVAL GUIDE)

Tài liệu này cung cấp hướng dẫn đầy đủ về yêu cầu hệ thống, cấu hình mô hình, cách thực thi và cấu trúc dữ liệu đầu ra khi sinh vector embedding cho corpus tin tức tiếng Việt nhằm phục vụ bài toán Retrieval-Augmented Generation (RAG).

---

## 1. Yêu cầu Hệ thống & Môi trường (Prerequisites & Requirements)

### Thư viện Python cần thiết
Đảm bảo bạn đã cài đặt đầy đủ các thư viện trong [requirements.txt](file:///Users/ai/Documents/HCMUS/Nam3/Ki3/Text%20Mining/Text-Mining---RAG-on-News/requirements.txt). Để chạy mô hình embedding cục bộ (local) hoặc trên Google Colab, cần cài đặt:
```bash
pip install sentence-transformers>=2.7.0 transformers>=4.51.0 accelerate numpy tqdm pandas datasets pyyaml python-dotenv
```
> [!NOTE]
> Mô hình `Qwen/Qwen3-Embedding-0.6B` yêu cầu các phiên bản `transformers` và `sentence-transformers` mới nhất để hỗ trợ các cơ chế attention mới và tham số `trust_remote_code=True`.

### Yêu cầu phần cứng
- **CPU**: Phù hợp cho việc chạy thử nghiệm nhỏ với số lượng bài viết hạn chế (`ARTICLE_LIMIT` thấp).
- **GPU (Khuyến nghị)**: CUDA GPU (ví dụ: NVIDIA T4 trên Colab) hoặc Apple Silicon MPS (đối với macOS). Embedding hàng nghìn bài viết thành hàng vạn chunk rất tốn tài nguyên tính toán nếu chạy bằng CPU.

---

## 2. Thông tin Mô hình Embedding & Cấu hình Chỉ dẫn (Model Configurations)

Hệ thống hỗ trợ cơ chế tự động nhận diện và cấu hình tiền tố chỉ dẫn (instruction-aware prefixes) thông qua hàm `detect_prefixes` trong [src/embed/embed_chunks.py](file:///Users/ai/Documents/HCMUS/Nam3/Ki3/Text%20Mining/Text-Mining---RAG-on-News/src/embed/embed_chunks.py).

### 2.1 Mô hình `intfloat/multilingual-e5-large`
- **Vector Dimension**: `1024`
- **Đặc trưng**: Đa ngôn ngữ tốt, hiệu năng cao trên tiếng Việt.
- **Cấu hình Prefixes**:
  - **Tài liệu/Chunk (Document Prefix)**: `"passage: "`
  - **Câu hỏi (Query Prefix)**: `"query: "`
- **Similarity Metric**: Normalize embeddings và dùng phép nhân vô hướng (Dot Product), tương đương với Cosine Similarity.

### 2.2 Mô hình `Qwen/Qwen3-Embedding-0.6B`
- **Vector Dimension**: Tự động nhận diện (1024 hoặc 1536).
- **Đặc trưng**: Mô hình instruction-aware mới, hỗ trợ tối đa 32k tokens ngữ cảnh.
- **Cấu hình Prefixes**:
  - **Tài liệu/Chunk (Document Prefix)**: `""` (Không dùng prefix/trống).
  - **Câu hỏi (Query Prefix)**:
    ```text
    Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: 
    ```
- **Lưu ý kỹ thuật đặc thù khi chạy Qwen3-Embedding**:
  1. **Tokenizer Padding Side**: Thiết lập `padding_side="left"` khi khởi tạo tokenizer của Qwen3 để tránh giảm sút chất lượng vector.
  2. **Giới hạn độ dài câu**: Đặt `model.max_seq_length = 512` để tránh tràn bộ nhớ GPU (CUDA Out of Memory) khi xử lý các chunk dài trong lô lớn (batch).
  3. **Cấu hình Attention**: Thiết lập `attn_implementation="eager"` (đặc biệt trong hàm [load_sentence_transformer](file:///Users/ai/Documents/HCMUS/Nam3/Ki3/Text%20Mining/Text-Mining---RAG-on-News/src/embed/embed_chunks.py#L343-L356)) để loại bỏ lỗi treo/nghẽn (deadlock) trong các môi trường chạy container như Google Colab.
  4. Cấu hình `trust_remote_code=True`.

---

## 3. Cấu trúc Dữ liệu Đầu vào (Input Data Format)

Dữ liệu đầu vào là các file JSONL chứa các chunk đã được phân tách từ Phase 3 (Chunking). Ví dụ: `src/chunking/output/vieonline_news_chunks_token.jsonl`.
Mỗi dòng trong file JSONL phải tuân theo cấu trúc:
```json
{
  "chunk_id": "chunk_unique_id",
  "article_id": "article_id",
  "text": "Nội dung văn bản gộp (tiêu đề + mô tả + nội dung chunk) dùng làm đầu vào embedding",
  "chunk_text": "Chỉ chứa nội dung thuần của chunk văn bản",
  "strategy": "token / langchain_recursive / llamaindex / structured",
  "metadata": {
    "title": "Tiêu đề bài báo",
    "category": "Thể loại bài báo",
    "chunk_index": 0,
    "token_count": 320
  }
}
```

---

## 4. Hướng dẫn Thực thi Sinh Embedding (How to Run)

### 4.1 Chạy qua CLI (Command Line Interface)
Sử dụng script [src/embed/embed_chunks.py](file:///Users/ai/Documents/HCMUS/Nam3/Ki3/Text%20Mining/Text-Mining---RAG-on-News/src/embed/embed_chunks.py) để sinh embedding cho từng strategy.

#### Ví dụ lệnh chạy với mô hình E5 mặc định:
```bash
python -m src.embed.embed_chunks \
  --input src/chunking/output/vieonline_news_chunks_token.jsonl \
  --output-dir src/embed/output/dense/token \
  --model intfloat/multilingual-e5-large \
  --batch-size 16 \
  --show-samples \
  --sample-size 5
```

#### Ví dụ lệnh chạy với mô hình Qwen3-Embedding:
```bash
python -m src.embed.embed_chunks \
  --input src/chunking/output/vieonline_news_chunks_token.jsonl \
  --output-dir src/embed/output/dense/token_qwen \
  --model Qwen/Qwen3-Embedding-0.6B \
  --batch-size 16 \
  --show-samples \
  --sample-size 5
```

Các tham số CLI chính:
- `--input`: Đường dẫn file chunk JSONL đầu vào (bắt buộc).
- `--output-dir`: Thư mục lưu kết quả embedding đầu ra.
- `--model`: Tên mô hình trên HuggingFace (mặc định là `intfloat/multilingual-e5-large`).
- `--batch-size`: Kích thước lô xử lý khi embed (mặc định là `16`). Giảm xuống nếu GPU yếu hoặc tăng lên để tăng tốc.
- `--show-samples`: In các đoạn mẫu preview của văn bản đầu vào sau khi embed để debug.
- `--no-progress`: Ẩn thanh tiến trình `tqdm`.
- `--no-normalize`: Không chuẩn hóa vector về độ dài bằng 1 (khuyến nghị giữ chuẩn hóa để tính Cosine Similarity nhanh hơn bằng Dot Product).
- `--document-prefix` và `--query-prefix`: Tự cấu hình thủ công tiền tố (mặc định chương trình tự nhận diện dựa vào tên mô hình).

### 4.2 Chạy qua Notebook tự động hóa pipeline
- **Chạy Local**: Mở file [src/embed/embedding_pipeline.ipynb](file:///Users/ai/Documents/HCMUS/Nam3/Ki3/Text%20Mining/Text-Mining---RAG-on-News/src/embed/embedding_pipeline.ipynb) để chạy tuần tự toàn bộ pipeline (Từ làm sạch dữ liệu -> Chunking -> So sánh -> Sinh Embedding -> Đánh giá sơ bộ QA).
- **Chạy Colab (Cho GPU T4 đơn)**: Sử dụng [src/embed/colab_embedding_pipeline.ipynb](file:///Users/ai/Documents/HCMUS/Nam3/Ki3/Text%20Mining/Text-Mining---RAG-on-News/src/embed/colab_embedding_pipeline.ipynb) để tải repo, cài đặt dependencies, chạy GPU sinh embedding cho mô hình Qwen3 một cách tối ưu nhất.
- **Chạy Kaggle (Cho GPU song song - GPU T4 x2 hoặc P100)**: Sử dụng [src/embed/kaggle_embedding_pipeline.ipynb](file:///Users/ai/Documents/HCMUS/Nam3/Ki3/Text%20Mining/Text-Mining---RAG-on-News/src/embed/kaggle_embedding_pipeline.ipynb) để tận dụng cấu hình 2 GPU song song. Hệ thống sẽ tự động phân phối các lô dữ liệu sang các GPU khác nhau thông qua multi-process GPU encoding pool giúp rút ngắn một nửa thời gian chạy.

> [!TIP]
> Để sử dụng song song 2 GPU trên Kaggle, hãy chọn **Accelerator: GPU T4 x2** ở menu cài đặt bên phải giao diện Kaggle Notebook, đồng thời bật **Internet: On** để có thể tải mô hình từ Hugging Face.

---

## 5. Cấu trúc dữ liệu đầu ra (Output Artifacts)

Sau khi chạy thành công, thư mục đầu ra sẽ chứa các tệp sau:
1. `embeddings.npy`: Ma trận biểu diễn vector `np.ndarray` lưu dưới định dạng nhị phân của NumPy (kiểu dữ liệu `float32`, kích thước `[N, Dimension]`).
2. `metadata.jsonl`: Danh sách metadata cho từng chunk văn bản, có thứ tự hàng trùng khớp hoàn toàn với các hàng trong `embeddings.npy`.
3. `manifest.json`: Lưu trữ thông tin tổng quan của phiên chạy (tên mô hình, tham số, prefixes được dùng, thời gian bắt đầu/kết thúc, số lượng vector).
4. `embedding_stats.json`: Chứa số liệu thống kê chi tiết (Thời gian xử lý, tốc độ chunks/sec, phân bố độ dài ký tự/token, danh sách 10 chunk dài nhất để kiểm soát lỗi trát ngữ cảnh).
5. `debug_samples.jsonl`: File mẫu để kiểm tra xem văn bản được định dạng chính xác với prefix tương ứng trước khi đưa vào mô hình hay chưa.

---

## 6. Truy xuất kiểm tra thử (Dense Search Verification)

Sau khi sinh embedding thành công, bạn có thể thực hiện tìm kiếm ngữ nghĩa thử nghiệm trực tiếp bằng script [src/embed/dense_search.py](file:///Users/ai/Documents/HCMUS/Nam3/Ki3/Text%20Mining/Text-Mining---RAG-on-News/src/embed/dense_search.py):

```bash
python -m src.embed.dense_search \
  --index-dir src/embed/output/dense/token \
  --query "Giá xăng dầu thế giới hôm nay biến động ra sao?" \
  --model intfloat/multilingual-e5-large \
  --top-k 5
```
> [!IMPORTANT]
> Mô hình truyền vào tham số `--model` phải trùng khớp với mô hình đã dùng để tạo index trong `--index-dir` để đảm bảo vector biểu diễn nằm trong cùng một không gian ngữ nghĩa.

Script tìm kiếm sẽ:
1. Tải file `manifest.json` trong thư mục index để lấy `query_prefix` phù hợp.
2. Mã hóa câu hỏi (Query) thành vector thông qua mô hình.
3. Thực hiện phép nhân ma trận (Dot Product) giữa vector truy vấn và toàn bộ vector trong `embeddings.npy`.
4. Trả về `top_k` chunk có điểm tương đồng cao nhất kèm metadata nguồn tương ứng (`chunk_id`, `article_id`, `score`, `text`...).

---

## 7. Các vấn đề thường gặp & Cách khắc phục (Troubleshooting)

- **CUDA Out of Memory (OOM)**:
  - *Nguyên nhân*: Chunk quá dài hoặc `batch_size` quá lớn.
  - *Khắc phục*: Giảm `--batch-size` xuống 8 hoặc 4. Đảm bảo model đã cấu hình `max_seq_length = 512` để tự động cắt tỉa văn bản quá dài.
- **Treo máy/Deadlock khi dùng Qwen trên Google Colab**:
  - *Nguyên nhân*: Lỗi tương thích attention của mô hình Qwen3 với thư viện transformers trên môi trường ảo hóa.
  - *Khắc phục*: Đảm bảo dùng tham số `attn_implementation="eager"` khi load model.
- **Lệch số lượng Vector và Metadata**:
  - *Nguyên nhân*: Quá trình ghi file bị gián đoạn giữa chừng.
  - *Khắc phục*: Xóa toàn bộ thư mục output của strategy đó và chạy lại.
- **Tìm kiếm không ra kết quả liên quan**:
  - *Nguyên nhân*: Dùng sai mô hình để mã hóa query so với mô hình sinh index, hoặc thiếu prefix bắt buộc đối với mô hình E5/Qwen.
  - *Khắc phục*: Đảm bảo truyền đúng tham số `--model` và kiểm tra lại `manifest.json` để xác nhận prefix đã được áp dụng tự động.
