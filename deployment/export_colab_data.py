"""Export portable RAG index artifacts for a Colab demo."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from src.backend import config


def create_bundle(output: Path, qdrant_path: Path, bm25_path: Path) -> Path:
    if not qdrant_path.is_dir():
        raise FileNotFoundError(f"Qdrant index not found: {qdrant_path}")
    if not bm25_path.is_file():
        raise FileNotFoundError(f"BM25 index not found: {bm25_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(qdrant_path.rglob("*")):
            if path.is_file() and path.name != ".lock":
                relative = path.relative_to(qdrant_path)
                archive.write(path, Path("data/qdrant_news") / relative)
        archive.write(bm25_path, "data/qdrant_news_bm25.pkl")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("rag_colab_data.zip"))
    args = parser.parse_args()
    result = create_bundle(args.output.resolve(), config.QDRANT_PATH, config.BM25_INDEX_PATH)
    print(f"Created {result} ({result.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
