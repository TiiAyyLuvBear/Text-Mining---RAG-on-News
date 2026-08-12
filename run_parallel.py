import subprocess
import threading
import sys
import time

configs = {
    "BGE_STRUCTURE": [
        ["python", "src/TEST_OUT/BGE_STRUCTURE/evaluate_llm_judge_structure.py",
         "--pred", "src/LLM_OUTPUT/BGE_STRUCTURE/answers_structure_bge_top5_claude.jsonl",
         "--output", "src/TEST_OUT/BGE_STRUCTURE/llm_judge_structure_bge_top5_claude_gptscore.jsonl",
         "--summary", "src/TEST_OUT/BGE_STRUCTURE/llm_judge_structure_bge_top5_claude_summary.json",
         "--resume"]
    ],
    "BGE_TOKEN": [
        ["python", "src/LLM_OUTPUT/BGE_TOKEN/generate_answers_claude_token.py",
         "--input", "src/re-ranker/out_reranker/rerank_token_bge_top5.jsonl",
         "--output", "src/LLM_OUTPUT/BGE_TOKEN/answers_token_bge_top5_claude.jsonl",
         "--resume"],
        ["python", "src/TEST_OUT/BGE_TOKEN/evaluate_auto_metrics.py",
         "--pred", "src/LLM_OUTPUT/BGE_TOKEN/answers_token_bge_top5_claude.jsonl",
         "--out-dir", "src/TEST_OUT/BGE_TOKEN"],
        ["python", "src/TEST_OUT/BGE_TOKEN/evaluate_llm_judge.py",
         "--pred", "src/LLM_OUTPUT/BGE_TOKEN/answers_token_bge_top5_claude.jsonl",
         "--output", "src/TEST_OUT/BGE_TOKEN/llm_judge_gptscore.jsonl",
         "--summary", "src/TEST_OUT/BGE_TOKEN/llm_judge_summary.json",
         "--resume"]
    ],
    "JINA_STRUCTURE": [
        ["python", "src/LLM_OUTPUT/JINA_STRUCTURE/jina_generate_answers_claude_structure.py",
         "--input", "src/re-ranker/out_reranker/rerank_structure_jina_top5.jsonl",
         "--output", "src/LLM_OUTPUT/JINA_STRUCTURE/answers_structure_jina_top5_claude.jsonl",
         "--resume"],
        ["python", "src/TEST_OUT/JINA_STRUCTURE/evaluate_auto_metrics_jina_structure.py",
         "--pred", "src/LLM_OUTPUT/JINA_STRUCTURE/answers_structure_jina_top5_claude.jsonl",
         "--out-dir", "src/TEST_OUT/JINA_STRUCTURE"],
        ["python", "src/TEST_OUT/JINA_STRUCTURE/evaluate_llm_judge_jina_structure.py",
         "--pred", "src/LLM_OUTPUT/JINA_STRUCTURE/answers_structure_jina_top5_claude.jsonl",
         "--output", "src/TEST_OUT/JINA_STRUCTURE/llm_judge_jina_structure_gptscore.jsonl",
         "--summary", "src/TEST_OUT/JINA_STRUCTURE/llm_judge_jina_structure_summary.json",
         "--resume"]
    ],
    "JINA_TOKEN": [
        ["python", "src/LLM_OUTPUT/JINA_TOKEN/jina_generate_answers_claude_token.py",
         "--input", "src/re-ranker/out_reranker/rerank_token_jina_top5.jsonl",
         "--output", "src/LLM_OUTPUT/JINA_TOKEN/answers_token_jina_top5_claude.jsonl",
         "--resume"],
        ["python", "src/TEST_OUT/JINA_TOKEN/evaluate_auto_metrics_jina_token.py",
         "--pred", "src/LLM_OUTPUT/JINA_TOKEN/answers_token_jina_top5_claude.jsonl",
         "--out-dir", "src/TEST_OUT/JINA_TOKEN"],
        ["python", "src/TEST_OUT/JINA_TOKEN/evaluate_llm_judge_jina_token.py",
         "--pred", "src/LLM_OUTPUT/JINA_TOKEN/answers_token_jina_top5_claude.jsonl",
         "--output", "src/TEST_OUT/JINA_TOKEN/llm_judge_jina_token_gptscore.jsonl",
         "--summary", "src/TEST_OUT/JINA_TOKEN/llm_judge_jina_token_summary.json",
         "--resume"]
    ]
}

def run_sequence(config_name, cmd_list):
    print(f"[{config_name}] Started sequence containing {len(cmd_list)} steps.")
    for idx, cmd in enumerate(cmd_list):
        print(f"[{config_name}] Step {idx+1}/{len(cmd_list)}: Running {' '.join(cmd)}")
        start_time = time.time()
        # Run process and stream output to console with prefix
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            sys.stdout.write(f"[{config_name}] {line}")
            sys.stdout.flush()
        process.wait()
        elapsed = time.time() - start_time
        if process.returncode != 0:
            print(f"[{config_name}] Step {idx+1} failed with code {process.returncode}. Stopping sequence.")
            return
        print(f"[{config_name}] Step {idx+1} completed in {elapsed:.2f} seconds.")
    print(f"[{config_name}] All steps completed successfully.")

threads = []
for config_name, cmd_list in configs.items():
    t = threading.Thread(target=run_sequence, args=(config_name, cmd_list), name=config_name)
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("All parallel pipeline processes have completed.")
