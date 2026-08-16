# Vietnamese News QA API

Pipeline:

```text
React/FastAPI -> E5-large -> Qdrant local -> BGE reranker -> OpenAI-compatible LLM API
```

## Setup

```powershell
pip install -r requirements.txt
Copy-Item .env.sample .env
# Set ANTHROPIC_API_KEY in .env
```

## Build the index

The default input is the existing token-chunked corpus at
`src/embed/output/chunks/vieonline_news_chunks_token.jsonl`:

```powershell
python -m src.qa_api.build_index
```

The first run downloads the E5 and BGE models and writes the persistent Qdrant index to `data/qdrant_news`.

## Run the API

```powershell
python -m src.qa_api.app
```

The FastAPI process serves both the API and the production React build on
`http://localhost:8000`. Use one process/worker because Qdrant is embedded.

- `GET /api/health`
- `POST /api/qa/ask`
- `WS /api/qa/stream`
- `WS /chat/stream` (legacy frontend compatibility)
- `POST /ask` (temporary legacy alias)

Example request:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/qa/ask `
  -ContentType 'application/json' `
  -Body '{"question":"Những loại thực phẩm nào nên hạn chế để giảm axit uric?"}'
```

## Build and serve React

```powershell
cd src/frontend
npm ci
npm run build
cd ../..
python -m src.qa_api.app
```

Vite development uses a proxy from `/api/*` to FastAPI port 8000. For a public
demo after building React, expose only the unified server:

```powershell
ngrok http 8000
```

Do not expose the embedded Qdrant directory or run Uvicorn with multiple workers.
