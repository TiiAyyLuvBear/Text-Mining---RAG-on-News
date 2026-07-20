from pathlib import Path
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

CHUNK_PATH = ROOT / os.getenv(
    "NEWS_CHUNK_PATH", "src/chunking/output/vieonline_news_chunks_token.jsonl"
)
QDRANT_PATH = ROOT / os.getenv("QDRANT_PATH", "data/qdrant_news")
COLLECTION = os.getenv("QDRANT_COLLECTION", "news_jina_token")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual"
)
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "claude-opus-4.8")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "20"))
TOP_K_CONTEXT = int(os.getenv("TOP_K_CONTEXT", "5"))
