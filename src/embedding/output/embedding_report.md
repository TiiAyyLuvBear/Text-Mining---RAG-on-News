# Embedding Experiment Report

- Model: `Alibaba-NLP/gte-multilingual-base`
- Answerable queries with gold articles: 114
- Relevance: article-level (top-k chunks mapped to article_id)
- Ranking metric: nDCG@10 (tie-break: Recall@10, then lower latency)

## Leaderboard

| Rank | Strategy | nDCG@10 | Recall@10 | Recall@5 | MRR@10 | Hit@1 | Hit@5 | Latency avg ms | Index MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | structured | 0.6181 | 0.6842 | 0.6754 | 0.5956 | 0.5263 | 0.6754 | 4.48 | 65.49 |
| 2 | llamaindex | 0.6181 | 0.6842 | 0.6754 | 0.5956 | 0.5263 | 0.6754 | 4.59 | 65.49 |
| 3 | token | 0.6169 | 0.6930 | 0.6842 | 0.5912 | 0.5175 | 0.6842 | 4.00 | 62.85 |
| 4 | langchain_recursive | 0.5902 | 0.6140 | 0.6140 | 0.5819 | 0.5526 | 0.6140 | 25.88 | 295.57 |

## Efficiency & Index

| Strategy | num_chunks | avg_chunk_tokens | embed_dim | embed_time_s | chunks/s | index_MB |
|---|---:|---:|---:|---:|---:|---:|
| token | 21454 | 428.2 | 768 | 871.57 | 24.6 | 62.85 |
| langchain_recursive | 100887 | 148.9 | 768 | 1801.33 | 56.0 | 295.57 |
| llamaindex | 22353 | 428.4 | 768 | 933.40 | 23.9 | 65.49 |
| structured | 22353 | 428.4 | 768 | 931.74 | 24.0 | 65.49 |
