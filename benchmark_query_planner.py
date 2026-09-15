import json
import time
from collections import Counter
from pathlib import Path
from tqdm import tqdm

from src.backend.query_planner import build_evidence_plan


def benchmark_planner(input_filepath: str, output_filepath: str):
    """Run lightweight heuristic benchmark strictly for query_planner.py."""
    input_path = Path(input_filepath)
    output_path = Path(output_filepath)

    if not input_path.exists():
        print(f"Dataset not found at {input_path}")
        return

    # Load dataset
    dataset = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                dataset.append(json.loads(line))

    print(f"Loaded {len(dataset)} questions. Benchmarking query_planner...")

    start_total_time = time.perf_counter()

    operator_counter = Counter()
    qa_type_counter = Counter()
    subquestion_count_distribution = Counter()
    suspicious_queries = []
    results = []

    for item in tqdm(dataset, desc="Analyzing Queries"):
        qid = item.get("id", "")
        question = item.get("question", "")
        qa_type = item.get("qa_type", "")

        qa_type_counter[qa_type] += 1

        t0 = time.perf_counter()
        plan = build_evidence_plan(question)
        latency_ms = (time.perf_counter() - t0) * 1000

        operator = plan.answer_operator
        sub_questions = [sq.text for sq in plan.sub_questions]
        num_sub_q = len(sub_questions)

        operator_counter[operator] += 1
        subquestion_count_distribution[num_sub_q] += 1

        # Heuristic checks for anomalies:
        # 1. Multi-doc comparison types that genuinely require >= 2 sub-questions
        # 2. Sub-questions with unresolved bracket indicators like "(đối tượng:"
        is_suspicious = False
        anomaly_reason = []

        # Only cross-document comparisons require multiple sub-questions; single-doc event summaries remain unified
        if qa_type in {"multi_doc_comparison"} and num_sub_q < 2 and operator != "DIRECT":
            is_suspicious = True
            anomaly_reason.append("Multi-doc comparison produced only 1 sub-question")

        if any("(đối tượng:" in sq for sq in sub_questions):
            is_suspicious = True
            anomaly_reason.append("Contains fallback marker '(đối tượng:'")

        record = {
            "id": qid,
            "qa_type": qa_type,
            "question": question,
            "operator": operator,
            "query_type_hint": plan.query_type_hint,
            "estimated_sources": plan.estimated_sources_needed,
            "entities": plan.entities,
            "dates": plan.dates,
            "numbers": plan.numbers,
            "temporal_constraints": plan.temporal_constraints,
            "sub_questions": sub_questions,
            "latency_ms": round(latency_ms, 3),
            "is_suspicious": is_suspicious,
            "anomaly_reason": "; ".join(anomaly_reason),
        }
        results.append(record)
        if is_suspicious:
            suspicious_queries.append(record)

    total_time = time.perf_counter() - start_total_time

    # Save detailed output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Print summary report
    print("\n" + "=" * 60)
    print("QUERY PLANNER BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total Questions Evaluated : {len(dataset)}")
    print(f"Total Execution Time      : {total_time:.3f} seconds")
    print(f"Average Latency           : {(total_time / len(dataset)) * 1000:.3f} ms/query")
    print(f"Throughput                : {len(dataset) / total_time:.1f} queries/second")
    print("-" * 60)
    print("Answer Operator Distribution:")
    for op, count in operator_counter.items():
        print(f"  - {op:<16}: {count:>3} ({count/len(dataset)*100:.1f}%)")
    print("-" * 60)
    print("Sub-questions Count Distribution:")
    for count, freq in sorted(subquestion_count_distribution.items()):
        print(f"  - {count} sub-question(s) : {freq:>3} queries ({freq/len(dataset)*100:.1f}%)")
    print("-" * 60)
    print(f"Suspicious / Edge Queries : {len(suspicious_queries)}")
    for item in suspicious_queries[:5]:
        print(f"  [ID: {item['id']}] ({item['qa_type']} -> {item['operator']}) {item['question']}")
        print(f"    -> Flag: {item['anomaly_reason']}")
        print(f"    -> Generated: {item['sub_questions']}")
    if len(suspicious_queries) > 5:
        print(f"  ... and {len(suspicious_queries) - 5} more flagged in the output file.")
    print("=" * 60)
    print(f"Details saved to: {output_path.resolve()}")


if __name__ == "__main__":
    INPUT_FILE = r"D:\HCMUS\HOCTAP\Semesters\25-26HK3\KhaiThacDuLieuVanBan\Project\Text-Mining---RAG-on-News\Dataset\QA_Claude\QA_output.jsonl"
    OUTPUT_FILE = r"D:\HCMUS\HOCTAP\Semesters\25-26HK3\KhaiThacDuLieuVanBan\Project\Text-Mining---RAG-on-News\Dataset\QA_Claude\planner_benchmark_results.jsonl"

    benchmark_planner(INPUT_FILE, OUTPUT_FILE)