# Chunking Experiment Report

| Strategy | Articles | Chunks | Avg chunks/article | Avg tokens/chunk | Time (s) | Chunks/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| token | 10073 | 21454 | 2.1299 | 364.6506 | 1.915025 | 11202.9869 |
| langchain_recursive | 10073 | 85426 | 8.4807 | 84.4122 | 34.331382 | 2488.2773 |
| llamaindex | 10073 | 43038 | 4.2726 | 173.4374 | 17.661206 | 2436.8664 |
| structured | 10073 | 22353 | 2.2191 | 364.6953 | 3.251637 | 6874.3836 |

## Quick Analysis

- `token` is the baseline: fixed token windows with token overlap.
- `langchain_recursive` preserves larger text boundaries first, then falls back to smaller separators.
- `llamaindex` uses `SentenceSplitter` when installed, with an internal sentence-window fallback otherwise.
- `structured` respects article paragraph structure before applying sentence/token windows.

Use `chunking_summary.json` for exact metrics and each `vieonline_news_chunks_<strategy>.jsonl` file for per-chunk inspection.
