# Embedding Experiment Report

- Model: `Alibaba-NLP/gte-multilingual-base`
- Answerable queries with gold articles: 114
- Relevance: article-level (top-k chunks mapped to article_id)
- Ranking metric: nDCG@10 (tie-break: Recall@10, then lower latency)

## Leaderboard

| Rank | Strategy | nDCG@10 | Recall@10 | Recall@5 | MRR@10 | Hit@1 | Hit@5 | Latency avg ms | Index MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | structured | 0.7526 | 0.8246 | 0.8129 | 0.7762 | 0.6667 | 0.9123 | 4.49 | 65.49 |
| 2 | llamaindex | 0.7526 | 0.8246 | 0.8129 | 0.7762 | 0.6667 | 0.9123 | 4.51 | 65.49 |
| 3 | token | 0.7519 | 0.8333 | 0.8216 | 0.7732 | 0.6579 | 0.9211 | 3.93 | 62.85 |
| 4 | langchain_recursive | 0.6993 | 0.7164 | 0.7164 | 0.7515 | 0.6842 | 0.8246 | 23.09 | 295.57 |

## Efficiency & Index

| Strategy | num_chunks | avg_chunk_tokens | embed_dim | embed_time_s | chunks/s | index_MB |
|---|---:|---:|---:|---:|---:|---:|
| token | 21454 | 428.2 | 768 | 762.78 | 28.1 | 62.85 |
| langchain_recursive | 100887 | 148.9 | 768 | 1559.71 | 64.7 | 295.57 |
| llamaindex | 22353 | 428.4 | 768 | 812.63 | 27.5 | 65.49 |
| structured | 22353 | 428.4 | 768 | 812.65 | 27.5 | 65.49 |
