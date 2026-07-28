from __future__ import annotations

import pickle
from pathlib import Path

from .schema import read_jsonl, result_from_chunk
from .tokenize import tokenize_vietnamese, tokenizer_name


INDEX_FILENAME = "bm25.pkl"


def build_bm25_index(chunks_path: str | Path, index_dir: str | Path, *, show_progress: bool = False) -> dict[str, object]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:  # pragma: no cover - dependency error
        raise RuntimeError("rank-bm25 is required; install it before building a BM25 index.") from exc

    rows = [row for row in read_jsonl(chunks_path) if str(row.get("text") or "").strip()]
    iterator = rows
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(rows, desc="Tokenizing for BM25", unit="chunk")
        except ImportError:  # pragma: no cover
            pass
    index = BM25Okapi([tokenize_vietnamese(str(row["text"])) for row in iterator])
    output = Path(index_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / INDEX_FILENAME).open("wb") as handle:
        pickle.dump({"index": index, "rows": rows, "tokenizer": tokenizer_name()}, handle)
    return {"index_path": str(output / INDEX_FILENAME), "chunks": len(rows), "tokenizer": tokenizer_name()}


def load_bm25_index(index_dir: str | Path) -> dict[str, object]:
    with (Path(index_dir) / INDEX_FILENAME).open("rb") as handle:
        return pickle.load(handle)


def search_bm25(index_data: dict[str, object], query: str, top_k: int) -> list[dict[str, object]]:
    index = index_data["index"]
    rows = index_data["rows"]
    scores = index.get_scores(tokenize_vietnamese(query))
    ordered = sorted(range(len(rows)), key=lambda position: (-float(scores[position]), str(rows[position]["chunk_id"])))[:top_k]
    return [result_from_chunk(rows[position], float(scores[position])) for position in ordered]
