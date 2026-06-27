# Notes - Tuấn Anh / BAAI/bge-m3

- Model: `BAAI/bge-m3`
- Retrieval: dense cosine/dot product over normalized embeddings.
- Query prefix/instruction: `(none)`
- Passage prefix/instruction: `(none)`
- Batch size: `32`
- Device: `cuda`
- Chunk size / overlap: `450` / `80`
- Total runtime seconds: `10644.420157`

Ranking chính dùng `nDCG@10`; khi gần nhau ưu tiên `Recall@10` cao hơn và latency thấp hơn.
