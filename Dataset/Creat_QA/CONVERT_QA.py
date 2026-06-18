import json
import csv

INPUT_JSONL  = "QA_output.jsonl"
OUTPUT_CSV   = "QA_output.csv"

def main():
    records = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("File JSONL rỗng!")
        return

    # Lấy tất cả columns từ record đầu tiên
    columns = ["id", "article_id", "question", "answers", "is_possible", "plausible_answers", "source_article_ids", "qa_type"]

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for rec in records:
            # answers và plausible_answers là list → convert sang string
            row = {**rec}
            row["answers"] = "; ".join(row.get("answers", []))
            row["plausible_answers"] = "; ".join(row.get("plausible_answers", []))
            # article_id và source_article_ids có thể là list (cross-article)
            if isinstance(row.get("article_id"), list):
                row["article_id"] = "; ".join(str(x) for x in row["article_id"])
            if isinstance(row.get("source_article_ids"), list):
                row["source_article_ids"] = "; ".join(str(x) for x in row["source_article_ids"])
            for col in columns:
                row.setdefault(col, "")
            writer.writerow(row)

    print(f"Đã chuyển {len(records)} records từ {INPUT_JSONL} → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
