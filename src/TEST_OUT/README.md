# TEST_OUT - QA Answer Evaluation

Thu muc nay dung de danh gia output RAG/LLM trong `src/LLM_OUTPUT/answers_token_bge_top5_claude.jsonl`.

## 1. Auto metrics

Metrics:

- BLEU
- ROUGE-L
- BERTScore Precision
- BERTScore Recall
- BERTScore F1

Cai dependencies:

```powershell
python -m pip install sacrebleu rouge-score bert-score
```

Chay danh gia:

```powershell
python src/TEST_OUT/evaluate_auto_metrics.py `
  --pred src/LLM_OUTPUT/answers_token_bge_top5_claude.jsonl `
  --gold Dataset/QA_Claude/QA_output.jsonl `
  --out-dir src/TEST_OUT
```

Neu may yeu hoac chua muon tai model BERTScore:

```powershell
python src/TEST_OUT/evaluate_auto_metrics.py --skip-bertscore
```

Output:

- `src/TEST_OUT/auto_metrics_summary.json`
- `src/TEST_OUT/auto_metrics_pairs.jsonl`

## 2. LLM-as-a-Judge / GPTScore

Dau vao judge gom:

- Cau hoi
- Ground Truth Answer
- Top-5 Context
- Generated Answer

LLM cham 5 tieu chi, moi tieu chi 1-5:

- Correctness
- Faithfulness
- Completeness
- Relevance
- Fluency

GPTScore = trung binh cong cua 5 tieu chi.

Set API key:

```powershell
$env:ANTHROPIC_API_KEY="ckey_api_key_cua_ban"
```

Chay thu 3 cau:

```powershell
python src/TEST_OUT/evaluate_llm_judge.py --limit 3
```

Chay toan bo:

```powershell
python src/TEST_OUT/evaluate_llm_judge.py
```

Chay tiep neu da co output:

```powershell
python src/TEST_OUT/evaluate_llm_judge.py --resume
```

Output:

- `src/TEST_OUT/llm_judge_gptscore.jsonl`
- `src/TEST_OUT/llm_judge_summary.json`
