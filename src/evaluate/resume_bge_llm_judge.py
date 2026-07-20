"""Check and resume missing BGE LLM-judge rows.

Usage:
    python src/evaluate/resume_bge_llm_judge.py --mode both --report
    python src/evaluate/resume_bge_llm_judge.py --mode token --resume
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = {
    "token": {
        "pred": ROOT / "src/generation/output/answers_token_bge_top5_claude.jsonl",
        "output": ROOT / "src/evaluate/bge_token/output/llm_judge_bge_token_gptscore.jsonl",
        "summary": ROOT / "src/evaluate/bge_token/output/llm_judge_bge_token_summary.json",
        "log": ROOT / "src/evaluate/bge_token/output/run.log",
        "judge": ROOT / "src/evaluate/jina_token/evaluate_llm_judge_jina_token.py",
        "report": ROOT / "src/evaluate/bge_token/output/resume_report.json",
    },
    "structure": {
        "pred": ROOT / "src/generation/output/answers_structure_bge_top5_claude.jsonl",
        "output": ROOT / "src/evaluate/bge_structure/output/llm_judge_bge_structure_gptscore.jsonl",
        "summary": ROOT / "src/evaluate/bge_structure/output/llm_judge_bge_structure_summary.json",
        "log": ROOT / "src/evaluate/bge_structure/output/run.log",
        "judge": ROOT / "src/evaluate/jina_structure/evaluation_llm_judge.py",
        "report": ROOT / "src/evaluate/bge_structure/output/resume_report.json",
    },
}


def ids(path: Path) -> list[str]:
    values = []
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("qa_id") is not None:
                values.append(str(row["qa_id"]))
    return values


def skipped_reasons(path: Path) -> dict[str, str]:
    reasons = {}
    if not path.exists():
        return reasons
    pattern = re.compile(r"Skipping qa_id (\S+) after retries: (.*)")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            reasons[match.group(1)] = match.group(2)
    return reasons


def inspect(name: str) -> dict:
    target = TARGETS[name]
    pred = ids(target["pred"])
    judged = ids(target["output"])
    pred_unique = list(dict.fromkeys(pred))
    judged_set = set(judged)
    missing = [qa_id for qa_id in pred_unique if qa_id not in judged_set]
    reasons = skipped_reasons(target["log"])
    report = {
        "mode": name,
        "prediction_rows": len(pred),
        "prediction_unique_ids": len(pred_unique),
        "judged_rows": len(judged),
        "judged_unique_ids": len(judged_set),
        "missing_ids": missing,
        "missing_count": len(missing),
        "skip_reasons": {qa_id: reasons.get(qa_id, "No skip entry found in log") for qa_id in missing},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    target["report"].parent.mkdir(parents=True, exist_ok=True)
    target["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def resume(name: str) -> int:
    target = TARGETS[name]
    command = [
        sys.executable,
        str(target["judge"]),
        "--pred", str(target["pred"]),
        "--gold", str(ROOT / "Dataset/QA_Claude/QA_output.jsonl"),
        "--output", str(target["output"]),
        "--summary", str(target["summary"]),
        "--resume",
    ]
    print("Running resume:", " ".join(command))
    return subprocess.run(command, cwd=ROOT).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["token", "structure", "both"], default="both")
    parser.add_argument("--resume", action="store_true", help="Resume API judge for missing IDs after reporting them.")
    args = parser.parse_args()
    names = ["token", "structure"] if args.mode == "both" else [args.mode]
    reports = [inspect(name) for name in names]
    if args.resume:
        for name, report in zip(names, reports):
            if report["missing_count"]:
                code = resume(name)
                if code:
                    raise SystemExit(code)
            else:
                print(f"[{name}] complete; no API calls needed")
        print("\nPost-resume check:")
        for name in names:
            inspect(name)


if __name__ == "__main__":
    main()
