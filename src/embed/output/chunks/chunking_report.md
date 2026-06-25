# Chunking Experiment Report

| Strategy | Articles | Chunks | Avg chunks/article | Avg tokens/chunk | Time (s) | Chunks/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| token | 10 | 24 | 2.4 | 343.0417 | 0.002385 | 10062.8931 |
| langchain_recursive | 10 | 121 | 12.1 | 75.686 | 0.003735 | 32396.2517 |
| llamaindex | 10 | 26 | 2.6 | 336.4615 | 0.003483 | 7464.8292 |
| structured | 10 | 26 | 2.6 | 336.4615 | 0.00346 | 7514.4509 |

## Quick Analysis

- `token` is the baseline: fixed token windows with token overlap.
- `langchain_recursive` preserves larger text boundaries first, then falls back to smaller separators.
- `llamaindex` uses `SentenceSplitter` when installed, with an internal sentence-window fallback otherwise.
- `structured` respects article paragraph structure before applying sentence/token windows.

Use `chunking_summary.json` for exact metrics and each `vieonline_news_chunks_<strategy>.jsonl` file for per-chunk inspection.
