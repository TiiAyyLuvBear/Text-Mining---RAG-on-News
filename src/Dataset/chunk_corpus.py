import argparse
import json
import os
import re

import pandas as pd
import tiktoken


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CSV_DIR = os.path.join(
    PROJECT_ROOT, "Dataset", "Create_QA_Vietonline", "VietOnlineNews_CSV"
)
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "Dataset", "chunks")
COLUMNS = ["id", "title", "description", "content", "category"]
ENC = tiktoken.get_encoding("cl100k_base")


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).replace("\ufeff", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_tokens(text):
    return len(ENC.encode(text or ""))


def split_sentences(text):
    text = clean_text(text)
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def token_window(text, max_tokens, overlap):
    tokens = ENC.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    step = max_tokens - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than max_tokens")

    start = 0
    while start < len(tokens):
        window = tokens[start : start + max_tokens]
        chunks.append(ENC.decode(window).strip())
        if start + max_tokens >= len(tokens):
            break
        start += step
    return [chunk for chunk in chunks if chunk]


def chunk_text(text, max_tokens=512, overlap=96, min_tokens=50):
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if sentence_tokens > max_tokens:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_tokens = 0
            chunks.extend(token_window(sentence, max_tokens, overlap))
            continue

        if current and current_tokens + sentence_tokens > max_tokens:
            chunk = " ".join(current).strip()
            chunks.append(chunk)

            overlap_sentences = []
            overlap_tokens = 0
            for old_sentence in reversed(current):
                old_tokens = count_tokens(old_sentence)
                if overlap_tokens + old_tokens > overlap:
                    break
                overlap_sentences.insert(0, old_sentence)
                overlap_tokens += old_tokens

            current = overlap_sentences + [sentence]
            current_tokens = overlap_tokens + sentence_tokens
        else:
            current.append(sentence)
            current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current).strip())

    filtered = [chunk for chunk in chunks if count_tokens(chunk) >= min_tokens]
    if not filtered and chunks:
        filtered = [max(chunks, key=count_tokens)]
    return filtered


def build_embedding_text(title, description, passage):
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if description:
        parts.append(f"Description: {description}")
    parts.append(f"Content: {passage}")
    return "\n".join(parts)


def read_csv_chunks(path, chunksize):
    return pd.read_csv(
        path,
        chunksize=chunksize,
        encoding="utf-8",
        dtype={
            "id": "Int64",
            "title": str,
            "description": str,
            "content": str,
            "category": str,
        },
    )


def process_split(split, csv_dir, out_dir, max_tokens, overlap, min_tokens, batch_rows):
    csv_path = os.path.join(csv_dir, f"{split}.csv")
    if not os.path.exists(csv_path):
        print(f"[SKIP] khong thay {csv_path}")
        return

    out_path = os.path.join(out_dir, f"{split}_chunks.jsonl")
    n_articles = 0
    n_chunks = 0
    chunk_token_sum = 0
    skipped_empty = 0

    print(f"[*] Chunking {split} tu CSV ...")
    with open(out_path, "w", encoding="utf-8") as fout:
        for df in read_csv_chunks(csv_path, batch_rows):
            for column in COLUMNS:
                if column not in df.columns:
                    df[column] = None

            for _, row in df.iterrows():
                if pd.isna(row["id"]):
                    continue

                article_id = int(row["id"])
                title = clean_text(row["title"])
                description = clean_text(row["description"])
                content = clean_text(row["content"])
                category = clean_text(row["category"])

                passages = chunk_text(content, max_tokens, overlap, min_tokens)
                if not passages:
                    skipped_empty += 1
                    continue

                n_articles += 1
                for chunk_index, passage in enumerate(passages):
                    n_tokens = count_tokens(passage)
                    chunk_token_sum += n_tokens
                    record = {
                        "chunk_id": f"article_{article_id}_chunk_{chunk_index}",
                        "article_id": article_id,
                        "title": title,
                        "description": description,
                        "category": category,
                        "source": "VietOnlineNews",
                        "url": "",
                        "chunk_index": chunk_index,
                        "n_tokens": n_tokens,
                        "text": passage,
                        "embedding_text": build_embedding_text(title, description, passage),
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_chunks += 1

    avg = chunk_token_sum / n_chunks if n_chunks else 0
    print(
        f"    bai={n_articles:,} chunks={n_chunks:,} "
        f"avg_tokens={avg:.1f} skipped_empty={skipped_empty:,}"
    )
    print(f"    -> {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default=DEFAULT_CSV_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--min-tokens", type=int, default=50)
    parser.add_argument("--batch-rows", type=int, default=5000)
    args = parser.parse_args()

    if args.overlap >= args.max_tokens:
        raise ValueError("--overlap phai nho hon --max-tokens")

    os.makedirs(args.out, exist_ok=True)
    for split in args.splits:
        process_split(
            split=split,
            csv_dir=args.csv_dir,
            out_dir=args.out,
            max_tokens=args.max_tokens,
            overlap=args.overlap,
            min_tokens=args.min_tokens,
            batch_rows=args.batch_rows,
        )


if __name__ == "__main__":
    main()



