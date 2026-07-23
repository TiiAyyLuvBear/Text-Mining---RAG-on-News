import argparse

from .pipeline import NewsPipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the local Qdrant news index.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Index only the first N chunks; useful for smoke tests.")
    args = parser.parse_args()

    pipeline = NewsPipeline()
    try:
        count = pipeline.build_index(batch_size=args.batch_size, limit=args.limit)
    finally:
        pipeline.close()
    print(f"Indexed {count} news chunks")
