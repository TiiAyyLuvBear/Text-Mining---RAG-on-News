from pathlib import Path
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

CHUNK_PATH = ROOT / os.getenv(
    "NEWS_CHUNK_PATH", "data/chunking/output/vieonline_news_chunks_token.jsonl"
)
QDRANT_PATH = ROOT / os.getenv("QDRANT_PATH", "data/qdrant_news")
COLLECTION = os.getenv("QDRANT_COLLECTION", "news_bge_token")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
)
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "claude-opus-4.8")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").strip().lower()
if LLM_PROVIDER not in {"auto", "api", "hf_model"}:
    raise ValueError("LLM_PROVIDER must be one of: auto, api, hf_model")
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "").strip()
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cuda:0").strip().lower()
HF_LLM_DEVICE = os.getenv("HF_LLM_DEVICE", MODEL_DEVICE).strip().lower()
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
LLM_API_URL = os.getenv(
    "LLM_API_URL",
    f"{ANTHROPIC_BASE_URL.rstrip('/')}/v1/chat/completions" if ANTHROPIC_BASE_URL else "https://api.xah.io/v1/chat/completions",
)
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2200"))
HF_LLM_MAX_NEW_TOKENS = int(os.getenv("HF_LLM_MAX_NEW_TOKENS", str(LLM_MAX_TOKENS)))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "90"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "20"))
TOP_K_CONTEXT = int(os.getenv("TOP_K_CONTEXT", "5"))
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "8"))
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "512"))
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "1.0"))
RERANK_MIN_MARGIN = float(os.getenv("RERANK_MIN_MARGIN", "2.0"))


def resolve_llm_provider() -> str:
    """Choose configured local Hugging Face generation or OpenAI-compatible API."""
    if LLM_PROVIDER == "auto":
        return "hf_model" if HF_LLM_MODEL else "api"
    return LLM_PROVIDER

