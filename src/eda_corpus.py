import os
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

COLUMNS = ["id", "title", "description", "content", "category"]


def analyze_and_convert(csv_path, parquet_path, chunksize=20000):
    n_rows = 0
    n_null = {c: 0 for c in COLUMNS}
    cat_counts = {}
    content_len_sum = 0
    content_len_min = None
    content_len_max = None
    seen_ids = set()
    dup_ids = 0
    empty_content = 0

    writer = None
    schema = pa.schema([
        ("id", pa.int64()),
        ("title", pa.string()),
        ("description", pa.string()),
        ("content", pa.string()),
        ("category", pa.string()),
    ])

    reader = pd.read_csv(csv_path, chunksize=chunksize, encoding="utf-8",
                         dtype={"title": str, "description": str, "content": str, "category": str})

    for chunk in reader:
        for c in COLUMNS:
            if c not in chunk.columns:
                chunk[c] = None
        chunk = chunk[COLUMNS]

        n_rows += len(chunk)
        for c in COLUMNS:
            n_null[c] += int(chunk[c].isna().sum())

        ids = chunk["id"].dropna().astype("int64")
        for i in ids:
            if i in seen_ids:
                dup_ids += 1
            else:
                seen_ids.add(i)

        cont = chunk["content"].fillna("").astype(str)
        lens = cont.str.len()
        content_len_sum += int(lens.sum())
        cmin = int(lens.min()) if len(lens) else 0
        cmax = int(lens.max()) if len(lens) else 0
        content_len_min = cmin if content_len_min is None else min(content_len_min, cmin)
        content_len_max = cmax if content_len_max is None else max(content_len_max, cmax)
        empty_content += int((lens == 0).sum())

        for cat, cnt in chunk["category"].fillna("(null)").value_counts().items():
            cat_counts[cat] = cat_counts.get(cat, 0) + int(cnt)

        chunk["id"] = pd.to_numeric(chunk["id"], errors="coerce").astype("Int64")
        table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, schema, compression="snappy")
        writer.write_table(table)

    if writer is not None:
        writer.close()

    avg_len = content_len_sum / n_rows if n_rows else 0
    return {
        "csv_path": csv_path,
        "parquet_path": parquet_path,
        "n_rows": n_rows,
        "n_unique_ids": len(seen_ids),
        "dup_ids": dup_ids,
        "n_null": n_null,
        "empty_content": empty_content,
        "content_len_min": content_len_min,
        "content_len_max": content_len_max,
        "content_len_avg": round(avg_len, 1),
        "category_counts": dict(sorted(cat_counts.items(), key=lambda x: -x[1])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join("Dataset", "VietOnlineNews_CSV"))
    ap.add_argument("--out-dir", default=os.path.join("Dataset", "parquet"))
    ap.add_argument("--chunksize", type=int, default=20000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    splits = ["train", "validation", "test"]

    all_stats = []
    for split in splits:
        csv_path = os.path.join(args.data_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            print(f"[SKIP] khong thay {csv_path}")
            continue
        parquet_path = os.path.join(args.out_dir, f"{split}.parquet")
        print(f"[*] Dang xu ly {split} ...")
        stats = analyze_and_convert(csv_path, parquet_path, args.chunksize)
        all_stats.append((split, stats))
        n_rows = stats["n_rows"]
        n_unique = stats["n_unique_ids"]
        dup = stats["dup_ids"]
        empty = stats["empty_content"]
        avg = stats["content_len_avg"]
        cmin = stats["content_len_min"]
        cmax = stats["content_len_max"]
        size_mb = os.path.getsize(parquet_path) / 1e6
        print(f"    rows={n_rows:,} unique_ids={n_unique:,} dup={dup} empty_content={empty}")
        print(f"    content_len avg={avg} min={cmin} max={cmax}")
        print(f"    -> {parquet_path} ({size_mb:.1f} MB)")

    print("\n" + "=" * 60)
    print("TONG KET EDA")
    print("=" * 60)
    for split, stats in all_stats:
        n_rows = stats["n_rows"]
        n_unique = stats["n_unique_ids"]
        dup = stats["dup_ids"]
        empty = stats["empty_content"]
        avg = stats["content_len_avg"]
        cmin = stats["content_len_min"]
        cmax = stats["content_len_max"]
        null_str = ", ".join(f"{k}={v}" for k, v in stats["n_null"].items())
        cat_counts = stats["category_counts"]
        print(f"\n### {split.upper()}")
        print(f"  So bai            : {n_rows:,}")
        print(f"  ID duy nhat       : {n_unique:,} (trung: {dup})")
        print(f"  Content rong      : {empty}")
        print(f"  Do dai content    : avg={avg} min={cmin} max={cmax} (ky tu)")
        print(f"  Null theo cot     : {null_str}")
        print(f"  So category       : {len(cat_counts)}")
        for cat, cnt in list(cat_counts.items())[:15]:
            print(f"      {cat:<25} {cnt:,}")


if __name__ == "__main__":
    main()
