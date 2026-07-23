# Hướng dẫn chạy Web Demo RAG

Tài liệu này hướng dẫn chạy backend và hai frontend của demo hỏi đáp tin tức tiếng Việt:

- Streamlit hiện tại: `src/frontend/app.py`.
- Giao diện pixel mới: `src/frontend/pixel.html`.
- Backend HTTP API: `src/backend/app.py`.

## 1. Chuẩn bị môi trường

Mở PowerShell tại thư mục gốc project. Project đã có Python virtual environment trong `Scripts`.

```powershell
.\Scripts\python.exe --version
.\Scripts\python.exe -m pip install -r requirements.txt
```

Kết quả mong đợi là Python 3.10.x.

## 2. Cấu hình API

Tạo hoặc cập nhật file `.env` ở thư mục gốc:

```env
ANTHROPIC_API_KEY=your_api_key
ANTHROPIC_MODEL=claude-opus-4.8
LLM_API_URL=https://api.xah.io/v1/chat/completions
LLM_MAX_TOKENS=900
RAG_API_HOST=127.0.0.1
RAG_API_PORT=8000
RAG_API_URL=http://127.0.0.1:8000/api/qa/ask
```

Không commit `.env` hoặc API key lên Git.

Backend dùng API OpenAI-compatible:

```http
POST https://api.xah.io/v1/chat/completions
Authorization: Bearer <ANTHROPIC_API_KEY>
```

`LLM_MAX_TOKENS` là giới hạn tối đa output. Nếu câu trả lời cần dài hơn, có thể tăng lên `1200` hoặc `1600`.

## 3. Chạy backend

Mở Terminal 1:

```powershell
.\Scripts\python.exe src\backend\app.py
```

Backend lắng nghe tại:

```text
http://127.0.0.1:8000/api/qa/ask
```

Log được rút gọn, ví dụ:

```text
15:47:28 | INFO | Backend listening on http://127.0.0.1:8000
15:47:28 | INFO | API POST /ask -> 200 (1250 ms, qa_id=211640_3)
```

## 4. Chạy Streamlit frontend

Mở Terminal 2:

```powershell
.\Scripts\python.exe -m streamlit run src\frontend\app.py
```

Mở `http://localhost:8501`.

Trong sidebar chọn:

- `Offline cache`: không cần backend hoặc API key; dùng output JSONL có sẵn.
- `Backend API`: cần backend đang chạy và API key hợp lệ.

## 5. Chạy frontend pixel

Mở Terminal 2 hoặc Terminal 3:

```powershell
.\Scripts\python.exe -m http.server 8502 --directory src\frontend
```

Mở:

```text
http://127.0.0.1:8502/pixel.html
```

Frontend pixel dùng Tailwind CDN và gọi `POST http://127.0.0.1:8000/api/qa/ask` bằng JavaScript. Vì vậy backend phải chạy trước khi bấm `Chạy truy vấn`.

## 6. Kiểm tra backend không gọi LLM

Kiểm tra CORS/OPTIONS:

```powershell
Invoke-WebRequest `
  -Uri http://127.0.0.1:8000/api/qa/ask `
  -Method Options `
  -UseBasicParsing
```

Kiểm tra input rỗng:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/qa/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body "{}"
```

Kết quả mong đợi là lỗi `400` với `Missing question`.

## 7. Kiểm tra câu hỏi thật

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/qa/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body (@{
    question = "Tại sao nước dùng hầm xương hoặc nước lẩu lại có thể gây hại cho thận?"
    top_k = 5
  } | ConvertTo-Json)
```

Request này sẽ gọi LLM bên ngoài và có thể phát sinh chi phí API.

## 8. Xử lý lỗi thường gặp

### `Connection refused` tới `127.0.0.1:8000`

Backend chưa chạy hoặc đang dùng cổng khác. Kiểm tra Terminal 1 và `RAG_API_PORT`.

### `401 Unauthorized` từ LLM provider

Kiểm tra API key và header dạng `Authorization: Bearer ...`. Không dùng key rỗng.

### `ConnectTimeout` hoặc `ReadTimeout`

Kiểm tra Internet, endpoint `LLM_API_URL`, model và tăng `ANTHROPIC_TIMEOUT` nếu provider phản hồi chậm.

### Câu trả lời quá ngắn

Kiểm tra `LLM_MAX_TOKENS` trong `.env`. Backend mặc định dùng `900`; sau khi sửa `.env` phải restart backend.

### Frontend pixel không có style

Frontend tải Tailwind từ CDN, nên cần Internet. JavaScript tùy chỉnh vẫn nằm trong `pixel.html`.
