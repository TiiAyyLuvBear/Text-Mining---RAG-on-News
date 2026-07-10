import argparse
import json
from pathlib import Path
from statistics import mean


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def load_gold(path):
    rows = {}
    for row in read_jsonl(path):
        qa_id = str(row.get("id"))
        answers = row.get("answers") or row.get("plausible_answers") or []
        rows[qa_id] = {
            "qa_id": qa_id,
            "question": row.get("question", ""),
            "reference_answer": answers[0] if answers else "",
            "answers": answers,
            "is_possible": row.get("is_possible"),
            "qa_type": row.get("qa_type"),
        }
    return rows


def load_pairs(pred_path, gold_path):
    gold_by_id = load_gold(gold_path)
    pairs = []
    missing_gold = 0
    for pred in read_jsonl(pred_path):
        qa_id = str(pred.get("qa_id"))
        gold = gold_by_id.get(qa_id)
        if gold is None:
            missing_gold += 1
            continue
        pairs.append({
            "qa_id": qa_id,
            "question": pred.get("question") or gold.get("question", ""),
            "reference_answer": gold.get("reference_answer", ""),
            "generated_answer": pred.get("generated_answer", ""),
            "qa_type": pred.get("qa_type") or gold.get("qa_type"),
            "llm_model": pred.get("llm_model"),
            "top_n_context": pred.get("top_n_context"),
            "used_contexts": pred.get("used_contexts", []),
        })
    return pairs, missing_gold


def compute_bleu(predictions, references):
    try:
        import sacrebleu
    except ImportError as exc:
        raise SystemExit("Missing dependency: sacrebleu. Install with: pip install sacrebleu") from exc
    return sacrebleu.corpus_bleu(predictions, [references]).score


def compute_rouge_l(predictions, references):
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise SystemExit("Missing dependency: rouge-score. Install with: pip install rouge-score") from exc
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
    return mean(scores) if scores else 0.0


def compute_bertscore(predictions, references, model_type, batch_size):
    try:
        from bert_score import score
    except ImportError as exc:
        raise SystemExit("Missing dependency: bert-score. Install with: pip install bert-score") from exc
    precision, recall, f1 = score(
        predictions,
        references,
        model_type=model_type,
        lang="vi",
        verbose=True,
        batch_size=batch_size,
        rescale_with_baseline=False,
    )
    return {
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
        "f1": float(f1.mean().item()),
    }


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated QA answers with BLEU, ROUGE-L, and BERTScore.")
    parser.add_argument("--pred", default="src/NOT_RAG/answers_token_top5_claude_not_rag.jsonl")
    parser.add_argument("--gold", default="Dataset/QA_Claude/QA_output.jsonl")
    parser.add_argument("--out-dir", default="src/NOT_RAG")
    parser.add_argument("--bertscore-model", default="xlm-roberta-large")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-bertscore", action="store_true")
    args = parser.parse_args()

    pairs, missing_gold = load_pairs(args.pred, args.gold)
    if not pairs:
        raise SystemExit("No matched prediction/gold pairs found.")

    predictions = [row["generated_answer"] for row in pairs]
    references = [row["reference_answer"] for row in pairs]

    per_question = []
    for row in pairs:
        per_question.append({
            "qa_id": row["qa_id"],
            "qa_type": row.get("qa_type"),
            "question": row["question"],
            "reference_answer": row["reference_answer"],
            "generated_answer": row["generated_answer"],
        })

    summary = {
        "num_predictions": sum(1 for line in Path(args.pred).open("r", encoding="utf-8") if line.strip()),
        "num_matched": len(pairs),
        "num_missing_gold": missing_gold,
        "bleu": compute_bleu(predictions, references),
        "rouge_l": compute_rouge_l(predictions, references),
    }

    if not args.skip_bertscore:
        bertscore = compute_bertscore(predictions, references, args.bertscore_model, args.batch_size)
        summary.update({
            "bertscore_model": args.bertscore_model,
            "bertscore_precision": bertscore["precision"],
            "bertscore_recall": bertscore["recall"],
            "bertscore_f1": bertscore["f1"],
        })

    out_dir = Path(args.out_dir)
    write_json(out_dir / "auto_metrics_not_rag_summary.json", summary)
    write_jsonl(out_dir / "auto_metrics_not_rag_pairs.jsonl", per_question)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
