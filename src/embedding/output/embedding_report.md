# Embedding Experiment Report

- Model: `Alibaba-NLP/gte-multilingual-base`
- Queries with gold articles: 152
- Relevance: article-level (top-k chunks mapped to article_id)
- Ranking metric: nDCG@10 (tie-break: Recall@10, then lower latency)

## Leaderboard

| Rank | Strategy | nDCG@10 | Recall@10 | Recall@5 | MRR@10 | Hit@1 | Hit@5 | Latency avg ms | Index MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | token | 0.5450 | 0.6118 | 0.6053 | 0.5226 | 0.4605 | 0.6053 | 4.40 | 62.85 |
| 2 | llamaindex | 0.5434 | 0.5987 | 0.5921 | 0.5246 | 0.4671 | 0.5921 | 4.08 | 65.49 |
| 3 | structured | 0.5434 | 0.5987 | 0.5921 | 0.5246 | 0.4671 | 0.5921 | 4.24 | 65.49 |
| 4 | langchain_recursive | 0.5291 | 0.5526 | 0.5526 | 0.5208 | 0.4934 | 0.5526 | 29.45 | 295.57 |

## Efficiency & Index

| Strategy | num_chunks | avg_chunk_tokens | embed_dim | embed_time_s | chunks/s | index_MB |
|---|---:|---:|---:|---:|---:|---:|
| token | 21454 | 428.2 | 768 | 894.78 | 24.0 | 62.85 |
| langchain_recursive | 100887 | 148.9 | 768 | 1871.40 | 53.9 | 295.57 |
| llamaindex | 22353 | 428.4 | 768 | 961.64 | 23.2 | 65.49 |
| structured | 22353 | 428.4 | 768 | 960.51 | 23.3 | 65.49 |
