import os
import json
import argparse
import pandas as pd
import pyarrow.parquet as pq


def load_corpus_ids(parquet_dir, splits):
    ids = set()
    per_split = {}
    for split in splits:
        path = os.path.join(parquet_dir, f"{split}.parquet")
        if not os.path.exists(path):
            print(f"[SKIP] khong thay {path}")
            continue
        table = pq.read_table(path, columns=["id"])
        split_ids = set(int(x) for x in table.column("id").to_pylist() if x is not None)
        per_split[split] = split_ids
        ids |= split_ids
        print(f"    {split}: {len(split_ids):,} ids")
    return ids, per_split


def load_qa(jsonl_path):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def norm_article_ids(rec):
    aid = rec.get("article_id")
    if aid is None:
        return []
    if isinstance(aid, list):
        return [int(x) for x in aid]
    return [int(aid)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=os.path.join("Dataset", "QA_Claude", "QA_output.jsonl"))
    ap.add_argument("--parquet-dir", default=os.path.join("Dataset", "parquet"))
    args = ap.parse_args()

    splits = ["train", "validation", "test"]
    print("[*] Doc corpus ids tu Parquet ...")
    corpus_ids, per_split = load_corpus_ids(args.parquet_dir, splits)
    print(f"    Tong unique ids corpus: {len(corpus_ids):,}")

    print("[*] Doc QA set ...")
    qa = load_qa(args.qa)
    n_qa = len(qa)

    n_possible = sum(1 for r in qa if r.get("is_possible") is True)
    n_impossible = sum(1 for r in qa if r.get("is_possible") is False)

    qa_types = {}
    for r in qa:
        t = r.get("qa_type") or "single-article"
        qa_types[t] = qa_types.get(t, 0) + 1

    missing = []
    referenced_ids = set()
    for r in qa:
        aids = norm_article_ids(r)
        for a in aids:
            referenced_ids.add(a)
            if a not in corpus_ids:
                missing.append((r.get("id"), a))

    empty_answers_possible = sum(
        1 for r in qa
        if r.get("is_possible") is True and not r.get("answers")
    )
    nonempty_answers_impossible = sum(
        1 for r in qa
        if r.get("is_possible") is False and r.get("answers")
    )

    print("\n" + "=" * 60)
    print("VALIDATE QA SET")
    print("=" * 60)
    print(f"  Tong cap QA            : {n_qa:,}")
    print(f"  is_possible = true     : {n_possible:,} ({n_possible/n_qa*100:.1f}%)")
    print(f"  is_possible = false    : {n_impossible:,} ({n_impossible/n_qa*100:.1f}%)")
    print(f"  Phan loai qa_type      :")
    for t, c in sorted(qa_types.items(), key=lambda x: -x[1]):
        print(f"      {t:<20} {c:,}")
    print(f"  Bai bao duoc tham chieu: {len(referenced_ids):,}")
    print(f"  article_id KHONG co trong corpus: {len(missing)}")
    for qid, aid in missing[:20]:
        print(f"      QA id={qid} -> article_id={aid} (missing)")
    if len(missing) > 20:
        print(f"      ... va {len(missing) - 20} truong hop khac")
    print(f"  [Check] is_possible=true nhung answers rong: {empty_answers_possible}")
    print(f"  [Check] is_possible=false nhung CO answers   : {nonempty_answers_impossible}")

    in_train = sum(1 for a in referenced_ids if a in per_split.get("train", set()))
    in_val = sum(1 for a in referenced_ids if a in per_split.get("validation", set()))
    in_test = sum(1 for a in referenced_ids if a in per_split.get("test", set()))
    print(f"  Bai tham chieu nam o   : train={in_train}, validation={in_val}, test={in_test}")


if __name__ == "__main__":
    main()
