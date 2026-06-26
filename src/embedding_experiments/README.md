# Embedding Experiment - Tuấn Anh / BAAI bge-m3

Chạy dense retrieval cho 4 chiến lược chunking:

- `token`
- `langchain_recursive`
- `llamaindex`
- `structured`

## Chạy nhanh trên subset

```powershell
python -m src.embedding_experiments.runner `
  --article-limit 200 `
  --query-limit 50 `
  --batch-size 8 `
  --device cpu
```

## Chạy full experiment

```powershell
python -m src.embedding_experiments.runner `
  --model BAAI/bge-m3 `
  --batch-size 16
```

Kết quả được lưu trong `reports/embedding_bge_m3/`:

- `leaderboard.csv`
- `leaderboard.md`
- `experiment_summary.json`
- `notes.md`
- `results_<strategy>.json`

`experiment_summary.json` có `stage_times.total_runtime_seconds` để báo cáo tổng thời gian chạy toàn bộ pipeline.
