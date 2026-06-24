import json
from pathlib import Path

PATH = Path(r"C:\Khai thác dữ liệu văn bản\Text-Mining---RAG-on-News\src\chunking\output\Fix QA - QA_output (1).jsonl")

records = []
with PATH.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        qa_type = record.pop("qa_type", "")
        record["qa_type"] = qa_type
        records.append(record)

with PATH.open("w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Done! Reordered {len(records)} lines -> {PATH}")
