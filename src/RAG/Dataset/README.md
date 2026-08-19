# Hướng Dẫn Các Script Trong `src`

Thư mục `src` có 4 file Python dùng để chuẩn bị dữ liệu cho bài toán Text Mining/RAG trên tin tức:

- `eda_corpus.py`: đọc dữ liệu CSV, phân tích nhanh corpus và chuyển sang Parquet.
- `chunk_corpus.py`: chia nội dung bài báo trong Parquet thành các đoạn nhỏ theo token.
- `validate_qa.py`: kiểm tra bộ câu hỏi-trả lời QA có khớp với corpus hay không.
- `_utf8_head.py`: cấu hình lại output terminal sang UTF-8 để in tiếng Việt ổn định hơn.

> Lưu ý quan trọng: các script đang dùng đường dẫn mặc định dạng `Dataset/...`, nên nên chạy từ thư mục gốc project, không chạy trực tiếp khi đang đứng trong `src`.
>
> Ví dụ đúng:
>
> ```powershell
> cd "C:\Khai thác dữ liệu văn bản\Text-Mining---RAG-on-News"
> python .\src\eda_corpus.py
> ```

---

## 1. `eda_corpus.py`

### Mục đích

File này dùng để:

1. Đọc dữ liệu tin tức từ các file CSV.
2. Kiểm tra nhanh chất lượng dữ liệu: số dòng, ID trùng, giá trị null, content rỗng, độ dài content, phân bố category.
3. Chuyển dữ liệu CSV sang định dạng Parquet để các bước sau đọc nhanh hơn.

### Input mặc định

Script tìm dữ liệu CSV trong thư mục:

```text
Dataset/VietOnlineNews_CSV
```

Các file được xử lý mặc định:

```text
Dataset/VietOnlineNews_CSV/train.csv
Dataset/VietOnlineNews_CSV/validation.csv
Dataset/VietOnlineNews_CSV/test.csv
```

Mỗi file CSV nên có các cột:

| Cột | Ý nghĩa |
|---|---|
| `id` | ID bài báo |
| `title` | Tiêu đề bài báo |
| `description` | Mô tả ngắn |
| `content` | Nội dung chính |
| `category` | Chuyên mục/tập nhãn |

Nếu thiếu cột nào trong 5 cột trên, script sẽ tự thêm cột đó với giá trị rỗng.

### Output mặc định

Script tạo các file Parquet trong:

```text
Dataset/parquet
```

Output gồm:

```text
Dataset/parquet/train.parquet
Dataset/parquet/validation.parquet
Dataset/parquet/test.parquet
```

### Cách chạy

Chạy từ thư mục gốc project:

```powershell
python .\src\eda_corpus.py
```

Nếu muốn chỉ định thư mục CSV và thư mục output:

```powershell
python .\src\eda_corpus.py --data-dir Dataset\VietOnlineNews_CSV --out-dir Dataset\parquet
```

Nếu file CSV lớn, có thể chỉnh số dòng đọc mỗi lần:

```powershell
python .\src\eda_corpus.py --chunksize 10000
```

### Tham số dòng lệnh

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `--data-dir` | `Dataset/VietOnlineNews_CSV` | Thư mục chứa `train.csv`, `validation.csv`, `test.csv` |
| `--out-dir` | `Dataset/parquet` | Thư mục lưu file Parquet |
| `--chunksize` | `20000` | Số dòng CSV đọc mỗi batch |

### Luồng xử lý chính

Hàm chính là:

```python
analyze_and_convert(csv_path, parquet_path, chunksize=20000)
```

Luồng xử lý:

1. Đọc CSV theo từng batch bằng `pandas.read_csv(..., chunksize=...)`.
2. Chuẩn hóa đủ 5 cột trong `COLUMNS`.
3. Đếm tổng số dòng.
4. Đếm null theo từng cột.
5. Kiểm tra ID trùng bằng `seen_ids`.
6. Tính độ dài `content`: min, max, trung bình.
7. Đếm số dòng có `content` rỗng.
8. Đếm phân bố `category`.
9. Chuyển batch sang bảng Arrow.
10. Ghi ra Parquet bằng `pyarrow.parquet.ParquetWriter` với nén `snappy`.

### Kết quả in ra màn hình

Ví dụ kết quả:

```text
[*] Dang xu ly train ...
    rows=100,000 unique_ids=100,000 dup=0 empty_content=12
    content_len avg=1540.5 min=0 max=12000
    -> Dataset/parquet/train.parquet (85.3 MB)

============================================================
TONG KET EDA
============================================================
```

Nếu thấy:

```text
[SKIP] khong thay Dataset\VietOnlineNews_CSV\train.csv
```

thì nguyên nhân thường là:

- Bạn đang chạy script từ sai thư mục, ví dụ đang đứng trong `src`.
- Dữ liệu CSV chưa nằm đúng vị trí.
- Tên thư mục hoặc tên file khác mặc định.

Cách sửa nhanh:

```powershell
cd "C:\Khai thác dữ liệu văn bản\Text-Mining---RAG-on-News"
python .\src\eda_corpus.py
```

Hoặc truyền đường dẫn tuyệt đối:

```powershell
python .\src\eda_corpus.py --data-dir "C:\Khai thác dữ liệu văn bản\Text-Mining---RAG-on-News\Dataset\VietOnlineNews_CSV"
```

---

## 2. `chunk_corpus.py`

### Mục đích

File này dùng để chia nội dung bài báo thành các đoạn nhỏ, gọi là `chunk` hoặc `passage`, phục vụ cho hệ thống RAG.

Trong RAG, không nên đưa cả bài báo dài vào vector database hoặc prompt. Thay vào đó, bài báo được chia thành nhiều đoạn ngắn hơn để:

- Dễ embedding hơn.
- Dễ truy xuất đúng đoạn liên quan.
- Giảm số token đưa vào mô hình.
- Tăng độ chính xác khi tìm kiếm ngữ nghĩa.

### Input mặc định

Script đọc các file Parquet từ:

```text
Dataset/parquet
```

Các file mặc định:

```text
Dataset/parquet/train.parquet
Dataset/parquet/validation.parquet
Dataset/parquet/test.parquet
```

Các cột được đọc:

```python
["id", "title", "description", "content", "category"]
```

### Output mặc định

Script tạo các file JSONL trong:

```text
Dataset/chunks
```

Output gồm:

```text
Dataset/chunks/train_chunks.jsonl
Dataset/chunks/validation_chunks.jsonl
Dataset/chunks/test_chunks.jsonl
```

Mỗi dòng JSONL là một chunk.

Ví dụ một record:

```json
{
  "chunk_id": "123_0",
  "article_id": 123,
  "title": "Tiêu đề bài báo",
  "category": "Thời sự",
  "url": "",
  "chunk_index": 0,
  "n_tokens": 350,
  "text": "Nội dung đoạn được cắt..."
}
```

### Cách chạy

Chạy từ thư mục gốc project:

```powershell
python .\src\chunk_corpus.py
```

Chỉ chạy cho một split:

```powershell
python .\src\chunk_corpus.py --splits train
```

Chạy cho nhiều split cụ thể:

```powershell
python .\src\chunk_corpus.py --splits train validation
```

Tùy chỉnh kích thước chunk:

```powershell
python .\src\chunk_corpus.py --max-tokens 384 --overlap 64
```

### Tham số dòng lệnh

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `--parquet-dir` | `Dataset/parquet` | Thư mục chứa file Parquet |
| `--out` | `Dataset/chunks` | Thư mục lưu file JSONL chunks |
| `--splits` | `train validation test` | Danh sách split cần xử lý |
| `--max-tokens` | `384` | Số token tối đa mỗi chunk |
| `--overlap` | `64` | Số token overlap giữa hai chunk liên tiếp |
| `--batch-rows` | `5000` | Số dòng Parquet đọc mỗi batch |

### Hàm `count_tokens`

```python
def count_tokens(text):
    return len(ENC.encode(text))
```

Hàm này dùng tokenizer `cl100k_base` của thư viện `tiktoken` để đếm số token trong văn bản.

Tokenizer này phù hợp với nhiều model OpenAI đời mới.

### Hàm `chunk_text`

```python
def chunk_text(text, max_tokens=384, overlap=64):
```

Hàm này nhận một đoạn text dài và trả về danh sách các chunk.

Cách hoạt động:

1. Xóa khoảng trắng đầu/cuối.
2. Nếu text rỗng thì trả về danh sách rỗng.
3. Encode text thành token.
4. Nếu số token nhỏ hơn hoặc bằng `max_tokens`, giữ nguyên text thành một chunk.
5. Nếu dài hơn, cắt theo cửa sổ token có kích thước `max_tokens`.
6. Mỗi cửa sổ sau lùi lại `overlap` token để giữ ngữ cảnh giữa các chunk.

Ví dụ với:

```text
max_tokens = 384
overlap = 64
```

thì bước nhảy giữa hai chunk là:

```text
384 - 64 = 320 token
```

Tức là:

- Chunk 1: token 0 đến 383.
- Chunk 2: token 320 đến 703.
- Chunk 3: token 640 đến 1023.

### Vì sao cần overlap?

Overlap giúp tránh mất ngữ cảnh ở ranh giới cắt đoạn.

Ví dụ một câu quan trọng nằm giữa cuối chunk 1 và đầu chunk 2. Nếu không overlap, hệ thống truy xuất có thể thiếu thông tin. Có overlap thì hai chunk cùng giữ một phần ngữ cảnh chung.

### Lưu ý

Trong docstring có ghi “cắt theo câu”, nhưng code hiện tại cắt trực tiếp theo token, không cắt theo dấu câu. Vì vậy chunk có thể bị cắt giữa câu.

Nếu muốn chunk mượt hơn, có thể cải tiến sau bằng cách tách câu trước rồi gom câu theo số token.

---

## 3. `validate_qa.py`

### Mục đích

File này dùng để kiểm tra bộ dữ liệu QA có hợp lệ so với corpus hay không.

Cụ thể script kiểm tra:

- Tổng số câu hỏi-trả lời.
- Số QA có thể trả lời được (`is_possible = true`).
- Số QA không thể trả lời được (`is_possible = false`).
- Phân bố loại câu hỏi `qa_type`.
- Các `article_id` được QA tham chiếu có tồn tại trong corpus hay không.
- QA possible nhưng không có answer.
- QA impossible nhưng lại có answer.
- Các bài báo được tham chiếu nằm ở split nào: train, validation, test.

### Input mặc định

QA file:

```text
Dataset/QA_Claude/QA_output.jsonl
```

Corpus Parquet:

```text
Dataset/parquet/train.parquet
Dataset/parquet/validation.parquet
Dataset/parquet/test.parquet
```

### Output

Script không tạo file mới. Kết quả được in ra terminal.

### Cách chạy

Chạy từ thư mục gốc project:

```powershell
python .\src\validate_qa.py
```

Chỉ định file QA khác:

```powershell
python .\src\validate_qa.py --qa Dataset\QA_Claude\QA_output.jsonl
```

Chỉ định thư mục Parquet khác:

```powershell
python .\src\validate_qa.py --parquet-dir Dataset\parquet
```

### Tham số dòng lệnh

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--qa` | `Dataset/QA_Claude/QA_output.jsonl` | Đường dẫn file QA JSONL |
| `--parquet-dir` | `Dataset/parquet` | Thư mục chứa corpus Parquet |

### Hàm `load_corpus_ids`

```python
def load_corpus_ids(parquet_dir, splits):
```

Hàm này:

1. Duyệt qua các split: `train`, `validation`, `test`.
2. Đọc cột `id` từ từng file Parquet.
3. Tạo tập ID riêng cho từng split.
4. Gộp toàn bộ ID vào `corpus_ids`.
5. Trả về:
   - `ids`: tập tất cả ID trong corpus.
   - `per_split`: dictionary chứa ID theo từng split.

### Hàm `load_qa`

```python
def load_qa(jsonl_path):
```

Hàm này đọc file JSONL, mỗi dòng là một object JSON.

Các dòng rỗng được bỏ qua.

### Hàm `norm_article_ids`

```python
def norm_article_ids(rec):
```

Hàm này chuẩn hóa trường `article_id` trong QA.

Vì `article_id` có thể là:

- Một số nguyên, ví dụ `123`.
- Một danh sách, ví dụ `[123, 456]`.
- Không có hoặc `None`.

Hàm sẽ luôn trả về list:

```python
123        -> [123]
[123,456]  -> [123, 456]
None       -> []
```

### Kết quả kiểm tra

Ví dụ output:

```text
VALIDATE QA SET
============================================================
  Tong cap QA            : 10,000
  is_possible = true     : 8,000 (80.0%)
  is_possible = false    : 2,000 (20.0%)
  Phan loai qa_type      :
      single-article       7,000
      multi-article        3,000
  Bai bao duoc tham chieu: 5,500
  article_id KHONG co trong corpus: 0
  [Check] is_possible=true nhung answers rong: 0
  [Check] is_possible=false nhung CO answers   : 0
  Bai tham chieu nam o   : train=4000, validation=1000, test=500
```

### Lưu ý lỗi chia cho 0

Trong code hiện tại có đoạn:

```python
n_possible/n_qa*100
n_impossible/n_qa*100
```

Nếu file QA rỗng, `n_qa = 0`, script sẽ lỗi chia cho 0. Vì vậy cần đảm bảo file QA có dữ liệu trước khi chạy.

---

## 4. `_utf8_head.py`

### Mục đích

File này cấu hình lại `stdout` và `stderr` sang UTF-8.

Nó hữu ích khi terminal Windows in tiếng Việt bị lỗi font hoặc lỗi encoding.

Nội dung chính:

```python
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
```

### Cách dùng

File này hiện chưa được import trong 3 script còn lại. Nếu muốn dùng, có thể thêm ở đầu mỗi script:

```python
import _utf8_head
```

Ví dụ trong `eda_corpus.py`:

```python
import _utf8_head
import os
import argparse
import pandas as pd
```

Khi đó các lệnh `print()` sẽ xuất ra UTF-8 ổn định hơn.

### Khi nào cần dùng?

Nên dùng nếu gặp các lỗi như:

- Tiếng Việt bị vỡ dấu.
- Terminal in ra ký tự lạ.
- PowerShell hoặc CMD báo lỗi encoding khi in text Unicode.

Ngoài ra, có thể chạy PowerShell với UTF-8 bằng:

```powershell
chcp 65001
```

---

## Quy Trình Chạy Đề Xuất

Nên chạy theo thứ tự sau từ thư mục gốc project:

### Bước 1: Chuyển CSV sang Parquet và xem EDA

```powershell
python .\src\eda_corpus.py
```

Kết quả mong đợi:

```text
Dataset/parquet/train.parquet
Dataset/parquet/validation.parquet
Dataset/parquet/test.parquet
```

### Bước 2: Chunk corpus cho RAG

```powershell
python .\src\chunk_corpus.py
```

Kết quả mong đợi:

```text
Dataset/chunks/train_chunks.jsonl
Dataset/chunks/validation_chunks.jsonl
Dataset/chunks/test_chunks.jsonl
```

### Bước 3: Validate QA

```powershell
python .\src\validate_qa.py
```

Kết quả mong đợi:

- Không có `article_id` bị missing.
- `is_possible=true` nên có `answers`.
- `is_possible=false` không nên có `answers`.

---

## Cấu Trúc Dữ Liệu Mong Đợi

Cấu trúc thư mục nên có dạng:

```text
Text-Mining---RAG-on-News/
├── Dataset/
│   ├── VietOnlineNews_CSV/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   ├── parquet/
│   │   ├── train.parquet
│   │   ├── validation.parquet
│   │   └── test.parquet
│   ├── chunks/
│   │   ├── train_chunks.jsonl
│   │   ├── validation_chunks.jsonl
│   │   └── test_chunks.jsonl
│   └── QA_Claude/
│       └── QA_output.jsonl
├── src/
│   ├── eda_corpus.py
│   ├── chunk_corpus.py
│   ├── validate_qa.py
│   ├── _utf8_head.py
│   └── README.md
└── README.md
```

---

## Lỗi Thường Gặp

### 1. `[SKIP] khong thay Dataset\...`

Nguyên nhân phổ biến nhất là chạy script khi đang đứng trong thư mục `src`.

Ví dụ đang ở:

```powershell
PS ...\Text-Mining---RAG-on-News\src>
```

mà chạy:

```powershell
python eda_corpus.py
```

thì script sẽ tìm:

```text
src/Dataset/VietOnlineNews_CSV/train.csv
```

trong khi dữ liệu thật thường nằm ở:

```text
Text-Mining---RAG-on-News/Dataset/VietOnlineNews_CSV/train.csv
```

Cách chạy đúng:

```powershell
cd "C:\Khai thác dữ liệu văn bản\Text-Mining---RAG-on-News"
python .\src\eda_corpus.py
```

Hoặc nếu vẫn muốn đứng trong `src`, truyền đường dẫn lùi một cấp:

```powershell
python .\eda_corpus.py --data-dir ..\Dataset\VietOnlineNews_CSV --out-dir ..\Dataset\parquet
```

### 2. Thiếu thư viện Python

Các script cần một số thư viện:

```text
pandas
pyarrow
tiktoken
```

Cài bằng:

```powershell
pip install pandas pyarrow tiktoken
```

### 3. File QA rỗng

Nếu `QA_output.jsonl` rỗng, `validate_qa.py` có thể lỗi do chia cho 0 khi tính phần trăm.

Cần kiểm tra file QA trước khi chạy validate.

---

## Tóm Tắt Nhanh

| File | Vai trò | Input chính | Output chính |
|---|---|---|---|
| `eda_corpus.py` | EDA CSV và chuyển CSV sang Parquet | `Dataset/VietOnlineNews_CSV/*.csv` | `Dataset/parquet/*.parquet` |
| `chunk_corpus.py` | Chia bài báo thành chunk theo token | `Dataset/parquet/*.parquet` | `Dataset/chunks/*_chunks.jsonl` |
| `validate_qa.py` | Kiểm tra QA có khớp corpus không | `Dataset/QA_Claude/QA_output.jsonl`, `Dataset/parquet/*.parquet` | In báo cáo ra terminal |
| `_utf8_head.py` | Cấu hình output UTF-8 | Không có | Không có, chỉ ảnh hưởng encoding terminal |
