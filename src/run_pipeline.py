import argparse
import subprocess
import sys
from pathlib import Path

# Paths to python files
PIPELINE_CONFIGS = {
    "BGE_STRUCTURE": {
        "generator": "src/LLM_OUTPUT/BGE_STRUCTURE/generate_answers_claude_structure.py",
        "eval_auto": "src/TEST_OUT/BGE_STRUCTURE/evaluate_auto_metrics_structure.py",
        "eval_judge": "src/TEST_OUT/BGE_STRUCTURE/evaluate_llm_judge_structure.py",
        "inputs": {
            "test": "src/re-ranker/out_reranker/rerank_structure_bge_top5_test.jsonl",
            "full": "src/re-ranker/out_reranker/rerank_structure_bge_top5.jsonl",
        },
        "outputs": {
            "test": {
                "gen": "src/LLM_OUTPUT/BGE_STRUCTURE/answers_structure_bge_top5_claude_test.jsonl",
                "eval_auto_dir": "src/TEST_OUT/BGE_STRUCTURE/test",
                "eval_judge_out": "src/TEST_OUT/BGE_STRUCTURE/llm_judge_structure_bge_top5_claude_gptscore_test.jsonl",
                "eval_judge_sum": "src/TEST_OUT/BGE_STRUCTURE/llm_judge_structure_bge_top5_claude_summary_test.json",
            },
            "full": {
                "gen": "src/LLM_OUTPUT/BGE_STRUCTURE/answers_structure_bge_top5_claude.jsonl",
                "eval_auto_dir": "src/TEST_OUT/BGE_STRUCTURE",
                "eval_judge_out": "src/TEST_OUT/BGE_STRUCTURE/llm_judge_structure_bge_top5_claude_gptscore.jsonl",
                "eval_judge_sum": "src/TEST_OUT/BGE_STRUCTURE/llm_judge_structure_bge_top5_claude_summary.json",
            }
        }
    },
    "BGE_TOKEN": {
        "generator": "src/LLM_OUTPUT/BGE_TOKEN/generate_answers_claude_token.py",
        "eval_auto": "src/TEST_OUT/BGE_TOKEN/evaluate_auto_metrics.py",
        "eval_judge": "src/TEST_OUT/BGE_TOKEN/evaluate_llm_judge.py",
        "inputs": {
            "test": "src/re-ranker/out_reranker/rerank_token_bge_top5_test.jsonl",
            "full": "src/re-ranker/out_reranker/rerank_token_bge_top5.jsonl",
        },
        "outputs": {
            "test": {
                "gen": "src/LLM_OUTPUT/BGE_TOKEN/answers_token_bge_top5_claude_test.jsonl",
                "eval_auto_dir": "src/TEST_OUT/BGE_TOKEN/test",
                "eval_judge_out": "src/TEST_OUT/BGE_TOKEN/llm_judge_gptscore_test.jsonl",
                "eval_judge_sum": "src/TEST_OUT/BGE_TOKEN/llm_judge_summary_test.json",
            },
            "full": {
                "gen": "src/LLM_OUTPUT/BGE_TOKEN/answers_token_bge_top5_claude.jsonl",
                "eval_auto_dir": "src/TEST_OUT/BGE_TOKEN",
                "eval_judge_out": "src/TEST_OUT/BGE_TOKEN/llm_judge_gptscore.jsonl",
                "eval_judge_sum": "src/TEST_OUT/BGE_TOKEN/llm_judge_summary.json",
            }
        }
    },
    "JINA_STRUCTURE": {
        "generator": "src/LLM_OUTPUT/JINA_STRUCTURE/jina_generate_answers_claude_structure.py",
        "eval_auto": "src/TEST_OUT/JINA_STRUCTURE/evaluate_auto_metrics_jina_structure.py",
        "eval_judge": "src/TEST_OUT/JINA_STRUCTURE/evaluate_llm_judge_jina_structure.py",
        "inputs": {
            "test": "src/re-ranker/out_reranker/rerank_structure_jina_top5_test.jsonl",
            "full": "src/re-ranker/out_reranker/rerank_structure_jina_top5.jsonl",
        },
        "outputs": {
            "test": {
                "gen": "src/LLM_OUTPUT/JINA_STRUCTURE/answers_structure_jina_top5_claude_test.jsonl",
                "eval_auto_dir": "src/TEST_OUT/JINA_STRUCTURE/test",
                "eval_judge_out": "src/TEST_OUT/JINA_STRUCTURE/llm_judge_jina_structure_gptscore_test.jsonl",
                "eval_judge_sum": "src/TEST_OUT/JINA_STRUCTURE/llm_judge_jina_structure_summary_test.json",
            },
            "full": {
                "gen": "src/LLM_OUTPUT/JINA_STRUCTURE/answers_structure_jina_top5_claude.jsonl",
                "eval_auto_dir": "src/TEST_OUT/JINA_STRUCTURE",
                "eval_judge_out": "src/TEST_OUT/JINA_STRUCTURE/llm_judge_jina_structure_gptscore.jsonl",
                "eval_judge_sum": "src/TEST_OUT/JINA_STRUCTURE/llm_judge_jina_structure_summary.json",
            }
        }
    },
    "JINA_TOKEN": {
        "generator": "src/LLM_OUTPUT/JINA_TOKEN/jina_generate_answers_claude_token.py",
        "eval_auto": "src/TEST_OUT/JINA_TOKEN/evaluate_auto_metrics_jina_token.py",
        "eval_judge": "src/TEST_OUT/JINA_TOKEN/evaluate_llm_judge_jina_token.py",
        "inputs": {
            "test": "src/re-ranker/out_reranker/rerank_token_jina_top5_test.jsonl",
            "full": "src/re-ranker/out_reranker/rerank_token_jina_top5.jsonl",
        },
        "outputs": {
            "test": {
                "gen": "src/LLM_OUTPUT/JINA_TOKEN/answers_token_jina_top5_claude_test.jsonl",
                "eval_auto_dir": "src/TEST_OUT/JINA_TOKEN/test",
                "eval_judge_out": "src/TEST_OUT/JINA_TOKEN/llm_judge_jina_token_gptscore_test.jsonl",
                "eval_judge_sum": "src/TEST_OUT/JINA_TOKEN/llm_judge_jina_token_summary_test.json",
            },
            "full": {
                "gen": "src/LLM_OUTPUT/JINA_TOKEN/answers_token_jina_top5_claude.jsonl",
                "eval_auto_dir": "src/TEST_OUT/JINA_TOKEN",
                "eval_judge_out": "src/TEST_OUT/JINA_TOKEN/llm_judge_jina_token_gptscore.jsonl",
                "eval_judge_sum": "src/TEST_OUT/JINA_TOKEN/llm_judge_jina_token_summary.json",
            }
        }
    }
}

def run_command(cmd):
    print(f"\nExecuting: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run LLM Answer Generation and Evaluations Pipeline")
    parser.add_argument("--mode", choices=["test", "full"], default="test", help="Run mode: test (5 rows) or full (152 rows)")
    parser.add_argument("--skip-bertscore", action="store_true", help="Skip BERTScore calculation in auto evaluations")
    parser.add_argument("--configs", nargs="+", default=list(PIPELINE_CONFIGS.keys()), help="Configs to run")
    args = parser.parse_args()

    print(f"=== Starting Pipeline Run ===")
    print(f"Mode: {args.mode.upper()}")
    print(f"Configs: {', '.join(args.configs)}")
    print(f"Skip BERTScore: {args.skip_bertscore}")
    print(f"=============================")

    for config_name in args.configs:
        if config_name not in PIPELINE_CONFIGS:
            print(f"Warning: Config {config_name} not found. Skipping.")
            continue

        print(f"\n>>> Running Pipeline for: {config_name} <<<")
        cfg = PIPELINE_CONFIGS[config_name]
        
        # Remove output files if they exist and we are not resuming
        if args.mode != "full":
            for path_key in ["gen", "eval_judge_out"]:
                p = Path(cfg["outputs"][args.mode][path_key])
                if p.exists():
                    print(f"Removing existing test file: {p}")
                    p.unlink()
        
        # 1. LLM Answer Generation
        input_file = cfg["inputs"][args.mode]
        output_file = cfg["outputs"][args.mode]["gen"]
        
        gen_cmd = [
            "python", cfg["generator"],
            "--input", input_file,
            "--output", output_file,
        ]
        if args.mode == "full":
            gen_cmd.append("--resume")
        run_command(gen_cmd)

        # 2. Automated Metrics Evaluation
        eval_auto_dir = cfg["outputs"][args.mode]["eval_auto_dir"]
        eval_auto_cmd = [
            "python", cfg["eval_auto"],
            "--pred", output_file,
            "--out-dir", eval_auto_dir,
        ]
        if args.skip_bertscore:
            eval_auto_cmd.append("--skip-bertscore")
        run_command(eval_auto_cmd)

        # 3. LLM-as-a-Judge Evaluation
        eval_judge_out = cfg["outputs"][args.mode]["eval_judge_out"]
        eval_judge_sum = cfg["outputs"][args.mode]["eval_judge_sum"]
        
        eval_judge_cmd = [
            "python", cfg["eval_judge"],
            "--pred", output_file,
            "--output", eval_judge_out,
            "--summary", eval_judge_sum,
        ]
        if args.mode == "full":
            eval_judge_cmd.append("--resume")
        run_command(eval_judge_cmd)

    print("\n=== Pipeline Run Completed Successfully ===")

if __name__ == "__main__":
    main()
