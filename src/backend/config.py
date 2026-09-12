from pathlib import Path
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def parse_cors_origins(value: str) -> tuple[str, ...]:
    """Parse a comma-separated CORS allow-list without accepting empty entries."""
    origins = tuple(origin.strip().rstrip("/") for origin in value.split(",") if origin.strip())
    return origins or ("http://localhost:5173", "http://127.0.0.1:5173")

CHUNK_PATH = ROOT / os.getenv(
    "NEWS_CHUNK_PATH", "data/chunking/output/vieonline_news_chunks_token.jsonl"
)
QDRANT_PATH = ROOT / os.getenv("QDRANT_PATH", "data/qdrant_news")
COLLECTION = os.getenv("QDRANT_COLLECTION", "news_bge_token")
BM25_INDEX_PATH = ROOT / os.getenv("BM25_INDEX_PATH", "data/qdrant_news_bm25.pkl")
REQUEST_TRACE_DIR = ROOT / os.getenv("REQUEST_TRACE_DIR", "logs/request_traces")
REQUEST_TRACE_ENABLED = os.getenv("REQUEST_TRACE_ENABLED", "true").strip().lower() in {
    "1", "true", "yes", "on",
}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
)
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "claude-opus-4.8")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "api").strip().lower()
if LLM_PROVIDER not in {"auto", "api", "hf_model"}:
    raise ValueError("LLM_PROVIDER must be one of: auto, api, hf_model")
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "CohereLabs/aya-expanse-8b").strip()
HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGING_FACE_HUB_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN")
    or os.getenv("HUGGING_FACE_ACCESS_KEY")
    or ""
).strip()
if HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", HF_TOKEN)
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cuda:0").strip().lower()
MODEL_DTYPE = os.getenv("MODEL_DTYPE", "float16").strip().lower()
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", MODEL_DEVICE).strip().lower()
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", MODEL_DEVICE).strip().lower()
HF_LLM_DEVICE = os.getenv("HF_LLM_DEVICE", MODEL_DEVICE).strip().lower()
HF_LLM_LOAD_IN_4BIT = os.getenv("HF_LLM_LOAD_IN_4BIT", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
HF_LLM_4BIT_QUANT_TYPE = os.getenv("HF_LLM_4BIT_QUANT_TYPE", "nf4").strip().lower()
HF_LLM_4BIT_USE_DOUBLE_QUANT = os.getenv("HF_LLM_4BIT_USE_DOUBLE_QUANT", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
HF_LLM_4BIT_COMPUTE_DTYPE = os.getenv("HF_LLM_4BIT_COMPUTE_DTYPE", "float16").strip().lower()
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
HYBRID_CANDIDATE_K = int(os.getenv("HYBRID_CANDIDATE_K", "50"))
HYBRID_RRF_K = int(os.getenv("HYBRID_RRF_K", "60"))
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "8"))
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "512"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "20"))
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "1.0"))
# BGE logits are ranking scores, not calibrated probabilities. These thresholds
# are conservative operating points until a labeled validation set exists.
RERANK_PARTIAL_MIN_SCORE = float(os.getenv("RERANK_PARTIAL_MIN_SCORE", "0.0"))
RERANK_EVIDENCE_CHUNK_DELTA = float(os.getenv("RERANK_EVIDENCE_CHUNK_DELTA", "1.5"))
RERANK_MIN_MARGIN = float(os.getenv("RERANK_MIN_MARGIN", "2.0"))
EVIDENCE_SUPPORT_THRESHOLD = float(os.getenv("EVIDENCE_SUPPORT_THRESHOLD", "0.70"))
EVIDENCE_LLM_ROUTING_ENABLED = os.getenv("EVIDENCE_LLM_ROUTING_ENABLED", "false").strip().lower() in {
    "1", "true", "yes", "on",
}
EVIDENCE_LLM_COVERAGE_ENABLED = os.getenv("EVIDENCE_LLM_COVERAGE_ENABLED", "false").strip().lower() in {
    "1", "true", "yes", "on",
}
# Slow-routing telemetry threshold. Provider-level LLM_TIMEOUT handles real
# request cancellation; a completed planner/scorer result is never discarded
# merely because the combined coverage pass exceeded this value.
EVIDENCE_PLAN_TIMEOUT = float(os.getenv("EVIDENCE_PLAN_TIMEOUT", "90"))
EVIDENCE_COVERAGE_CANDIDATE_K = int(os.getenv("EVIDENCE_COVERAGE_CANDIDATE_K", "10"))
EVIDENCE_ENTAILMENT_BATCH_SIZE = int(os.getenv("EVIDENCE_ENTAILMENT_BATCH_SIZE", "10"))
EVIDENCE_SUPPORT_TEXT_MAX_CHARS = int(os.getenv("EVIDENCE_SUPPORT_TEXT_MAX_CHARS", "1200"))
TEMPORAL_BOOST = float(os.getenv("TEMPORAL_BOOST", "0.05"))
SOURCE_MAX_CHUNKS_PER_ARTICLE = int(os.getenv("SOURCE_MAX_CHUNKS_PER_ARTICLE", "2"))
CORS_ORIGINS = parse_cors_origins(
    os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
)

# Generation-stage controls. These are deliberately independent of retrieval
# thresholds: upstream evidence routing owns answerability, while this stage
# only compresses and packages already-selected evidence.
CONTEXT_COMPRESSION_THRESHOLD = float(
    os.getenv("CONTEXT_COMPRESSION_THRESHOLD", "0.25")
)
CONTEXT_TOKEN_BUDGET = int(os.getenv("CONTEXT_TOKEN_BUDGET", "2500"))
CONTEXT_COMPRESSION_ENABLED = os.getenv(
    "CONTEXT_COMPRESSION_ENABLED", "false"
).strip().lower() in {"1", "true", "yes", "on"}


def resolve_llm_provider() -> str:
    """Choose configured local Hugging Face generation or OpenAI-compatible API."""
    if LLM_PROVIDER == "auto":
        return "hf_model" if HF_LLM_MODEL else "api"
    return LLM_PROVIDER
