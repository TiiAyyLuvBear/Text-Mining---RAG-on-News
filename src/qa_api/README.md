# Vietnamese News QA API

Pipeline:

```text
E5-large -> Qdrant local -> BGE reranker -> Claude
```

## Setup

```powershell
pip install -r requirements.txt
Copy-Item .env.sample .env
# Set ANTHROPIC_API_KEY in .env
```

## Build the index

The default input is the token-chunked news corpus:

```powershell
python -m src.qa_api.build_index
```

The first run downloads the E5 and BGE models and writes the persistent Qdrant index to `data/qdrant_news`.

## Run the API

```powershell
python -m src.qa_api.app
```

API: `http://localhost:8000`

- `GET /api/health`
- `POST /api/qa/ask`
- `WS /api/qa/stream`
- `WS /chat/stream` (legacy frontend compatibility)

Example request:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/qa/ask `
  -ContentType 'application/json' `
  -Body '{"question":"Những loại thực phẩm nào nên hạn chế để giảm axit uric?"}'
```

For a CPU smoke test without rebuilding the full corpus, use:

```powershell
python -m src.qa_api.build_index --limit 32 --batch-size 8
```

For a CPU smoke test without rebuilding the full corpus, use:

```powershell
python -m src.qa_api.build_index --limit 32 --batch-size 8
```
