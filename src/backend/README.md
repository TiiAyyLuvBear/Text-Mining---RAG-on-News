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
python -m src.backend.build_index
```

The first run downloads the E5 and BGE models and writes the persistent Qdrant index to `data/qdrant_news`.

## Run the API

```powershell
python -m src.backend.app
```

API: `http://localhost:8000`

## LLM provider

Configure generator in `.env`:

```dotenv
LLM_PROVIDER=auto
MODEL_DEVICE=cuda:0
HF_LLM_MODEL=
HF_LLM_DEVICE=cuda:0
HF_LLM_MAX_NEW_TOKENS=900
```

`auto` uses Hugging Face when `HF_LLM_MODEL` is set; otherwise it uses the OpenAI-compatible API. Use `api` or `hf_model` to force one provider. API mode reads `LLM_API_KEY`, `LLM_API_URL`, and `GENERATOR_MODEL`. Hugging Face mode expects a text-generation model and loads it lazily on first generated answer.

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
python -m src.backend.build_index --limit 32 --batch-size 8
```

Vite development uses a proxy from `/api/*` to FastAPI port 8000. For a public
demo after building React, expose only the unified server:

```powershell
python -m src.backend.build_index --limit 32 --batch-size 8
```

Do not expose the embedded Qdrant directory or run Uvicorn with multiple workers.
