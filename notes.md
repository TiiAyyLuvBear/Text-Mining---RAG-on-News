# Embedding Model Notes - Qwen3-Embedding

Tài liệu này ghi nhận cấu hình prompt và instruction của mô hình `Qwen/Qwen3-Embedding-0.6B` cho thực nghiệm Retrieval.

---

## 1. Thông Tin Mô Hình
- **Model ID**: `Qwen/Qwen3-Embedding-0.6B`
- **Thành viên phụ trách**: My
- **Đặc trưng**: Mô hình instruction-aware mới, hỗ trợ tối đa 32k tokens ngữ cảnh.

---

## 2. Cấu Hình Prompt & Instruction

### Cho Query (Câu hỏi truy vấn)
Mô hình yêu cầu một cấu trúc chỉ dẫn cụ thể (Task-specific Instruction) đặt trước nội dung câu hỏi để tối ưu khả năng tìm kiếm:
- **Cấu trúc**:
  ```text
  Instruct: Given a web search query, retrieve relevant passages that answer the query
  Query: {Nội dung câu hỏi}
  ```
- **Chuỗi tiền tố (Query Prefix)**:
  `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "`

### Cho Document (Tài liệu / Chunk)
Mô hình **không** yêu cầu tiền tố (prefix) hay instruction khi sinh embedding cho tài liệu/passage:
- **Chuỗi tiền tố (Document Prefix)**:
  `""` (Trống)

---

## 3. Lưu Ý Kỹ Thuật Khi Chạy
1. **Tokenizer Padding Side**: Phải thiết lập `padding_side="left"` khi load tokenizer của Qwen3 để tránh giảm sút chất lượng vector do đệm từ bên phải.
2. **Giới hạn độ dài câu**: Đặt tối đa `model.max_seq_length = 512` để tránh tràn bộ nhớ GPU (CUDA Out of Memory) khi xử lý các chunk dài trong lô lớn (batch).
3. **Cấu hình Attention**: Thiết lập `attn_implementation="eager"` để loại bỏ lỗi treo/nghẽn (deadlock) trong các môi trường chạy container như Google Colab.
