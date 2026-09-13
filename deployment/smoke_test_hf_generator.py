"""Load the configured local HF generator and produce one real Vietnamese answer."""

from __future__ import annotations

import csv
import json
import threading

from src.backend import config
from src.backend.pipeline import NewsPipeline


QUESTION = (
    "Những loại nội tạng động vật nào được khuyến cáo nên hạn chế để tránh "
    "tăng axit uric và hại thận?"
)


def _real_context() -> dict[str, object]:
    path = config.ROOT / "Dataset/Create_QA_Vietonline/VietOnlineNews/train_new.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        row = next(item for item in csv.DictReader(handle) if item["id"] == "211640")
    return {
        "article_id": "211640",
        "chunk_id": "211640-smoke",
        "citation_rank": 1,
        "text": " ".join(str(value) for value in row.values() if value),
    }


def main() -> int:
    import torch

    provider = config.resolve_llm_provider()
    if provider != "hf_model":
        raise RuntimeError(f"Smoke test requires LLM_PROVIDER=hf_model, got {provider!r}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Colab generator smoke test.")

    pipeline = NewsPipeline.__new__(NewsPipeline)
    pipeline.generator = None
    pipeline.generator_provider = provider
    pipeline._load_lock = threading.Lock()
    pipeline._generate_with_api = lambda _: (_ for _ in ()).throw(
        AssertionError("API generation must not run in hf_model mode")
    )
    answer = pipeline.generate(QUESTION, [_real_context()])
    if not answer.strip():
        raise RuntimeError("Hugging Face generator returned an empty answer.")

    model = pipeline.generator.model
    model_device = str(next(model.parameters()).device)
    loaded_in_4bit = bool(getattr(model, "is_loaded_in_4bit", False))
    if not model_device.startswith("cuda"):
        raise RuntimeError(f"Generator model is not on CUDA: {model_device}")
    if config.HF_LLM_LOAD_IN_4BIT and not loaded_in_4bit:
        raise RuntimeError("4-bit was requested but the loaded model does not report 4-bit quantization.")

    print(json.dumps({
        "provider": provider,
        "model": config.HF_LLM_MODEL,
        "device": model_device,
        "load_in_4bit": loaded_in_4bit,
        "answer": answer,
        "api_called": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
