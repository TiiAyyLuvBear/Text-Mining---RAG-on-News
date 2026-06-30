# Chunking Experiment Report

| Strategy | Articles | Chunks | Avg chunks/article | Avg tokens/chunk | Time (s) | Chunks/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| token | 10073 | 24099 | 2.3924 | 333.4086 | 1.34444 | 17924.9353 |
| langchain_recursive | 10073 | 98156 | 9.7445 | 74.0653 | 6.775508 | 14486.8842 |
| llamaindex | 10073 | 48977 | 4.8622 | 154.4036 | 10.307869 | 4751.4186 |
| structured | 10073 | 25464 | 2.5279 | 332.1436 | 2.725311 | 9343.5208 |

## Quick Analysis

- `token` is the baseline: fixed token windows with token overlap.
- `langchain_recursive` preserves larger text boundaries first, then falls back to smaller separators.
- `llamaindex` uses `SentenceSplitter` when installed, with an internal sentence-window fallback otherwise.
- `structured` respects article paragraph structure before applying sentence/token windows.

Use `chunking_summary.json` for exact metrics and each `vieonline_news_chunks_<strategy>.jsonl` file for per-chunk inspection.
