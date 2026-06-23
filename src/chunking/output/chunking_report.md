# Chunking Experiment Report

| Strategy | Articles | Chunks | Avg chunks/article | Avg tokens/chunk | Time (s) | Chunks/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| token | 10073 | 21454 | 2.1299 | 364.6506 | 3.145694 | 6820.1166 |
| langchain_recursive | 10073 | 85426 | 8.4807 | 84.4122 | 13.585894 | 6287.8453 |
| llamaindex | 10073 | 43038 | 4.2726 | 173.4374 | 23.882764 | 1802.0527 |
| structured | 10073 | 22353 | 2.2191 | 364.6953 | 5.418926 | 4124.9871 |

## Quick Analysis

- `token` is the baseline: fixed token windows with token overlap.
- `langchain_recursive` preserves larger text boundaries first, then falls back to smaller separators.
- `llamaindex` uses `SentenceSplitter` when installed, with an internal sentence-window fallback otherwise.
- `structured` respects article paragraph structure before applying sentence/token windows.

Use `chunking_summary.json` for exact metrics and each `vieonline_news_chunks_<strategy>.jsonl` file for per-chunk inspection.
