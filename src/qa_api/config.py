from pathlib import Path
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

CHUNK_PATH = ROOT / os.getenv(
    "NEWS_CHUNK_PATH", "src/chunking/output/vieonline_news_chunks_token.jsonl"
)
QDRANT_PATH = ROOT / os.getenv("QDRANT_PATH", "data/qdrant_news")
COLLECTION = os.getenv("QDRANT_COLLECTION", "news_bge_token")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
)
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "claude-opus-4.8")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
LLM_API_URL = os.getenv(
    "LLM_API_URL",
    f"{ANTHROPIC_BASE_URL.rstrip('/')}/v1/chat/completions" if ANTHROPIC_BASE_URL else "https://api.xah.io/v1/chat/completions",
)
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2200"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "90"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "20"))
TOP_K_CONTEXT = int(os.getenv("TOP_K_CONTEXT", "5"))
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "8"))
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "512"))
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "2.0"))
RERANK_MIN_MARGIN = float(os.getenv("RERANK_MIN_MARGIN", "2.0"))
