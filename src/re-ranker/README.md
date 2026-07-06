# BGE Reranker for Token Embedding

Script này rerank `src/embedding/output/per_query_token.jsonl` bằng model `BAAI/bge-reranker-v2-m3` và lấy top 5 context.

## Cài dependency

```bash
pip install FlagEmbedding
```

Nếu chạy GPU, nên cài PyTorch CUDA đúng môi trường trước.

## Test nhanh 5 câu

```bash
python src/re-ranker/bge_rerank_token.py --limit 5 --batch-size 4
```

Output mặc định:

```text
src/re-ranker/output/rerank_token_bge_top5.jsonl
```

## Chạy full

```bash
python src/re-ranker/bge_rerank_token.py --batch-size 8 --fp16
```

Nếu chạy CPU hoặc bị lỗi fp16, bỏ `--fp16`:

```bash
python src/re-ranker/bge_rerank_token.py --batch-size 4
```

## Input

```text
src/embedding/output/per_query_token.jsonl
```

Script dùng các trường:

- `qa_id`
- `qa_type`
- `question`
- `gold_articles`
- `candidates[*].rank`
- `candidates[*].chunk_index`
- `candidates[*].chunk_id`
- `candidates[*].article_id`
- `candidates[*].score`
- `candidates[*].text`

## Output

Mỗi dòng gồm:

- `qa_id`
- `question`
- `embedding_type = token`
- `reranker = BAAI/bge-reranker-v2-m3`
- `reranked_candidates`: top 5 đã sort theo `rerank_score`
- `rerank_metrics`: `hit@1`, `hit@5`, `recall@5`, `mrr@5`, `ndcg@5`
- `original_retrieval_metrics`: metric cũ từ embedding output
