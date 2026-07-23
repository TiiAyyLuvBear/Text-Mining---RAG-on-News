from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .pipeline import NewsPipeline

pipeline = NewsPipeline()


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        yield
    finally:
        pipeline.close()


app = FastAPI(title="Vietnamese News QA API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "index_ready": pipeline.is_ready()}


@app.get("/documents")
def documents():
    return {"documents": []}


@app.post("/api/qa/ask")
def ask(request: AskRequest):
    contexts = pipeline.search(request.question, request.top_k)
    answer = pipeline.generate(request.question, contexts)
    return {
        "answer": answer,
        "citations": [
            {
                "article_id": item.get("article_id"),
                "title": item.get("title"),
                "url": item.get("url"),
                "score": item.get("rerank_score"),
            }
            for item in contexts
        ],
        "retrieval": contexts,
    }


async def stream_answer(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            body = json.loads(await websocket.receive_text())
            question = str(body.get("text") or body.get("question") or "").strip()
            if not question:
                continue
            contexts = await asyncio.to_thread(pipeline.search, question, 5)
            answer = await asyncio.to_thread(pipeline.generate, question, contexts)
            for token in answer.split(" "):
                await websocket.send_text(token + " ")
                await asyncio.sleep(0.005)
    except WebSocketDisconnect:
        return


@app.websocket("/api/qa/stream")
async def qa_stream(websocket: WebSocket):
    await stream_answer(websocket)


@app.websocket("/chat/stream")
async def legacy_chat_stream(websocket: WebSocket):
    await stream_answer(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
