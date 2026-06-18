import os
import json
import argparse
import pandas as pd
import pyarrow.parquet as pq
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    return len(ENC.encode(text))


def chunk_text(text, max_tokens=384, overlap=64):
    """Cat text thanh cac passage theo token, co overlap.
    Cat theo cau (split tho bang dau cham) roi gom lai cho gan max_tokens.
    """
    text = (text or "").strip()
    if not text:
        return []

    tokens = ENC.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    start = 0
    step = max_tokens - overlap
    while start < len(tokens):
        window = tokens[start:start + max_tokens]
        chunks.append(ENC.decode(window))
        if start + max_tokens >= len(tokens):
            break
        start += step
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", default=os.path.join("Dataset", "parquet"))
    ap.add_argument("--out", default=os.path.join("Dataset", "chunks"))
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--batch-rows", type=int, default=5000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    for split in args.splits:
        path = os.path.join(args.parquet_dir, f"{split}.parquet")
        if not os.path.exists(path):
            print(f"[SKIP] khong thay {path}")
            continue

        out_path = os.path.join(args.out, f"{split}_chunks.jsonl")
        pf = pq.ParquetFile(path)
        n_articles = 0
        n_chunks = 0
        chunk_token_sum = 0

        print(f"[*] Chunking {split} ...")
        with open(out_path, "w", encoding="utf-8") as fout:
            for batch in pf.iter_batches(batch_size=args.batch_rows,
                                         columns=["id", "title", "description", "content", "category"]):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    aid = row["id"]
                    if pd.isna(aid):
                        continue
                    aid = int(aid)
                    title = str(row["title"]) if not pd.isna(row["title"]) else ""
                    category = str(row["category"]) if not pd.isna(row["category"]) else ""
                    content = str(row["content"]) if not pd.isna(row["content"]) else ""

                    passages = chunk_text(content, args.max_tokens, args.overlap)
                    n_articles += 1
                    for ci, passage in enumerate(passages):
                        ntok = count_tokens(passage)
                        chunk_token_sum += ntok
                        rec = {
                            "chunk_id": f"{aid}_{ci}",
                            "article_id": aid,
                            "title": title,
                            "category": category,
                            "url": "",
                            "chunk_index": ci,
                            "n_tokens": ntok,
                            "text": passage,
                        }
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_chunks += 1

        avg = chunk_token_sum / n_chunks if n_chunks else 0
        print(f"    bai={n_articles:,} chunks={n_chunks:,} avg_tokens={avg:.1f}")
        print(f"    -> {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
