from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - handled at runtime for optional chart output
    plt = None

# D:\HCMUS\HOCTAP\Semesters\25-26HK3\KhaiThacDuLieuVanBan\Project\Text-Mining---RAG-on-News\src\TEST_OUT\QWEN\data_QA_Convert.jsonl
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# DEFAULT_ARTICLES = PROJECT_ROOT / "Dataset" / "Create_QA_Vietonline" / "VietOnlineNews" / "train_new.jsonl"
DEFAULT_QA = PROJECT_ROOT / "TEST_OUT" / "QWEN" / "data_QA_Convert.jsonl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "EDA_output"

# TEXT_FIELDS = ("title", "description", "content")
MOJIBAKE_MARKERS = ("Ă", "Æ", "Ä", "áº", "á»", "â€", "Å")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(obj)
    return records


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(text_value(item) for item in value)
    return str(value).strip()


def token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def has_mojibake(text: str) -> bool:
    return any(marker in text for marker in MOJIBAKE_MARKERS)


def maybe_fix_mojibake(text: str) -> str:
    if not has_mojibake(text):
        return text
    try:
        fixed = text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text
    return fixed if len(fixed) >= max(1, len(text) // 2) else text


def quantiles(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"min": 0, "p25": 0, "mean": 0, "median": 0, "p75": 0, "p95": 0, "max": 0}
    series = pd.Series(values)
    return {
        "min": round(float(series.min()), 2),
        "p25": round(float(series.quantile(0.25)), 2),
        "mean": round(float(series.mean()), 2),
        "median": round(float(series.median()), 2),
        "p75": round(float(series.quantile(0.75)), 2),
        "p95": round(float(series.quantile(0.95)), 2),
        "max": round(float(series.max()), 2),
    }


# def article_id_set(records: list[dict[str, Any]]) -> set[str]:
#     ids: set[str] = set()
#     for record in records:
#         article_id = text_value(record.get("id"))
#         if article_id:
#             ids.add(article_id)
#     return ids


# def summarize_articles(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
#     rows: list[dict[str, Any]] = []
#     category_counter: Counter[str] = Counter()
#     id_counter: Counter[str] = Counter()
#     mojibake_records = 0
#
#     for record in records:
#         article_id = text_value(record.get("id"))
#         title = text_value(record.get("title"))
#         description = text_value(record.get("description"))
#         content = text_value(record.get("content"))
#         category = text_value(record.get("category")) or "missing"
#         joined_text = " ".join([title, description, content])
#
#         id_counter[article_id] += 1
#         category_counter[category] += 1
#         if has_mojibake(joined_text):
#             mojibake_records += 1
#
#         rows.append(
#             {
#                 "id": article_id,
#                 "category": category,
#                 "title_chars": len(title),
#                 "description_chars": len(description),
#                 "content_chars": len(content),
#                 "title_tokens": token_count(title),
#                 "description_tokens": token_count(description),
#                 "content_tokens": token_count(content),
#                 "has_title": bool(title),
#                 "has_description": bool(description),
#                 "has_content": bool(content),
#                 "has_mojibake": has_mojibake(joined_text),
#             }
#         )
#
#     article_df = pd.DataFrame(rows)
#     category_df = pd.DataFrame(category_counter.most_common(), columns=["category", "count"])
#     summary = {
#         "num_articles": len(records),
#         "unique_article_ids": len(set(id_counter)),
#         "duplicate_article_ids": sum(1 for _, count in id_counter.items() if count > 1),
#         "missing_title": int((~article_df["has_title"]).sum()) if not article_df.empty else 0,
#         "missing_description": int((~article_df["has_description"]).sum()) if not article_df.empty else 0,
#         "missing_content": int((~article_df["has_content"]).sum()) if not article_df.empty else 0,
#         "mojibake_records": mojibake_records,
#         "num_categories": len(category_counter),
#         "content_char_stats": quantiles(article_df["content_chars"].tolist() if not article_df.empty else []),
#         "content_token_stats": quantiles(article_df["content_tokens"].tolist() if not article_df.empty else []),
#     }
#     return article_df, category_df, summary


def normalize_article_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [text_value(item) for item in value if text_value(item)]
    text = text_value(value)
    return [text] if text else []


def summarize_qa(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    qa_type_counter: Counter[str] = Counter()
    possible_counter: Counter[str] = Counter()
    qa_id_counter: Counter[str] = Counter()
    question_counter: Counter[str] = Counter()

    for record in records:
        qa_id = text_value(record.get("id"))
        article_ids = normalize_article_ids(record.get("article_id"))
        question = text_value(record.get("question"))
        answers = record.get("answers") if isinstance(record.get("answers"), list) else []
        plausible_answers = record.get("plausible_answers") if isinstance(record.get("plausible_answers"), list) else []
        qa_type = text_value(record.get("qa_type")) or "missing"
        is_possible = record.get("is_possible")
        possible_label = "missing" if is_possible is None else str(bool(is_possible))
        answer_text = text_value(answers)
        plausible_text = text_value(plausible_answers)
        joined_text = " ".join([question, answer_text, plausible_text])

        qa_id_counter[qa_id] += 1
        question_counter[question] += 1
        qa_type_counter[qa_type] += 1
        possible_counter[possible_label] += 1

        rows.append(
            {
                "id": qa_id,
                "article_ids": ",".join(article_ids),
                "qa_type": qa_type,
                "is_possible": possible_label,
                "num_answers": len(answers),
                "num_plausible_answers": len(plausible_answers),
                "question_chars": len(question),
                "answer_chars": len(answer_text),
                "question_tokens": token_count(question),
                "answer_tokens": token_count(answer_text),
                "has_question": bool(question),
                "has_answer": bool(answer_text),
                "has_mojibake": has_mojibake(joined_text),
            }
        )

    qa_df = pd.DataFrame(rows)
    qa_type_df = pd.DataFrame(qa_type_counter.most_common(), columns=["qa_type", "count"])

    summary = {
        "num_qa": len(records),
        "unique_qa_ids": len(set(qa_id_counter)),
        "duplicate_qa_ids": sum(1 for _, count in qa_id_counter.items() if count > 1),
        "duplicate_questions": sum(1 for question, count in question_counter.items() if question and count > 1),
        "missing_questions": int((~qa_df["has_question"]).sum()) if not qa_df.empty else 0,
        "missing_answers": int((~qa_df["has_answer"]).sum()) if not qa_df.empty else 0,
        "mojibake_qa_records": int(qa_df["has_mojibake"].sum()) if not qa_df.empty else 0,
        "qa_type_distribution": dict(qa_type_counter),
        "is_possible_distribution": dict(possible_counter),
        "question_token_stats": quantiles(qa_df["question_tokens"].tolist() if not qa_df.empty else []),
        "answer_token_stats": quantiles(qa_df["answer_tokens"].tolist() if not qa_df.empty else []),
    }
    return qa_df, qa_type_df, summary


def percentage(part: int | float, total: int | float) -> float:
    if not total:
        return 0.0
    return round(float(part) / float(total) * 100, 2)


def make_markdown_report(
    qa_summary: dict[str, Any],
    qa_type_df: pd.DataFrame,
) -> str:
    lines = [
        "# EDA Report",
        "",
        "## QA Set",
        "",
        f"- QA pairs: {qa_summary['num_qa']:,}",
        f"- Unique QA ids: {qa_summary['unique_qa_ids']:,}",
        f"- Duplicate QA ids: {qa_summary['duplicate_qa_ids']:,}",
        f"- Duplicate questions: {qa_summary['duplicate_questions']:,}",
        f"- Missing questions: {qa_summary['missing_questions']:,}",
        f"- Missing answers: {qa_summary['missing_answers']:,}",
        f"- Records with possible encoding/mojibake issues: {qa_summary['mojibake_qa_records']:,} ({percentage(qa_summary['mojibake_qa_records'], qa_summary['num_qa'])}%)",
        "",
        "### Question Length Stats",
        "",
        pd.DataFrame([qa_summary["question_token_stats"]], index=["tokens"]).to_markdown(),
        "",
        "### Answer Length Stats",
        "",
        pd.DataFrame([qa_summary["answer_token_stats"]], index=["tokens"]).to_markdown(),
        "",
        "### QA Type Distribution",
        "",
        qa_type_df.to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)


def save_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_path: Path,
    top_n: int | None = None,
) -> None:
    if plt is None or df.empty:
        return

    plot_df = df.head(top_n).copy() if top_n else df.copy()
    plot_df = plot_df.sort_values(y_col, ascending=True)

    height = max(4.5, min(9.0, 0.45 * len(plot_df) + 1.5))
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(plot_df[x_col].astype(str), plot_df[y_col], color="#2f6f9f")
    ax.set_title(title)
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    for index, value in enumerate(plot_df[y_col]):
        ax.text(value, index, f" {int(value):,}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_histogram(
    values: pd.Series,
    title: str,
    xlabel: str,
    output_path: Path,
    bins: int = 40,
) -> None:
    if plt is None or values.empty:
        return

    clean_values = pd.to_numeric(values, errors="coerce").dropna()
    if clean_values.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(clean_values, bins=bins, color="#4f8f6f", edgecolor="white")
    ax.axvline(
        clean_values.mean(),
        color="#c2410c",
        linestyle="--",
        linewidth=1.8,
        label=f"Mean: {clean_values.mean():.2f}",
    )
    ax.axvline(
        clean_values.median(),
        color="#7c3aed",
        linestyle="--",
        linewidth=1.8,
        label=f"Median: {clean_values.median():.2f}",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_boxplot(
    data: list[tuple[str, pd.Series]],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if plt is None:
        return

    labels: list[str] = []
    values: list[list[float]] = []

    for label, series in data:
        clean_values = pd.to_numeric(series, errors="coerce").dropna().tolist()
        if clean_values:
            labels.append(label)
            values.append(clean_values)

    if not values:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    try:
        ax.boxplot(
            values,
            tick_labels=labels,
            patch_artist=True,
            boxprops={"facecolor": "#8fb9a8"},
        )
    except TypeError:
        ax.boxplot(
            values,
            labels=labels,
            patch_artist=True,
            boxprops={"facecolor": "#8fb9a8"},
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_charts(
    qa_df: pd.DataFrame,
    qa_type_df: pd.DataFrame,
    out_dir: Path,
) -> list[Path]:
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    if plt is None:
        return []

    chart_paths = [
        charts_dir / "qa_type_distribution.png",
        charts_dir / "qa_question_tokens_hist.png",
        charts_dir / "qa_answer_tokens_hist.png",
        charts_dir / "qa_possible_distribution.png",
        charts_dir / "qa_question_answer_boxplot.png",
    ]

    save_bar_chart(
        qa_type_df,
        "qa_type",
        "count",
        "QA Type Distribution",
        chart_paths[0],
    )

    save_histogram(
        qa_df["question_tokens"],
        "Question Length Distribution",
        "Question tokens",
        chart_paths[1],
        bins=25,
    )

    save_histogram(
        qa_df["answer_tokens"],
        "Answer Length Distribution",
        "Answer tokens",
        chart_paths[2],
        bins=25,
    )

    possible_df = (
        qa_df["is_possible"]
        .value_counts()
        .rename_axis("is_possible")
        .reset_index(name="count")
    )

    save_bar_chart(
        possible_df,
        "is_possible",
        "count",
        "QA is_possible Distribution",
        chart_paths[3],
    )

    save_boxplot(
        [
            ("Question", qa_df["question_tokens"]),
            ("Answer", qa_df["answer_tokens"]),
        ],
        "Question vs Answer Length",
        "Tokens",
        chart_paths[4],
    )

    return [path for path in chart_paths if path.exists()]


def print_summary(qa_summary: dict[str, Any]) -> None:
    print("=" * 72)
    print("QA SET")
    print("=" * 72)
    print(f"QA pairs                  : {qa_summary['num_qa']:,}")
    print(f"Unique QA ids             : {qa_summary['unique_qa_ids']:,}")
    print(f"Duplicate questions       : {qa_summary['duplicate_questions']:,}")
    print(f"Missing answers           : {qa_summary['missing_answers']:,}")
    print(f"Possible mojibake records : {qa_summary['mojibake_qa_records']:,}")
    print(f"Question tokens mean/p95  : {qa_summary['question_token_stats']['mean']} / {qa_summary['question_token_stats']['p95']}")
    print(f"Answer tokens mean/p95    : {qa_summary['answer_token_stats']['mean']} / {qa_summary['answer_token_stats']['p95']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EDA for QA JSONL files."
    )
    # parser.add_argument(
    #     "--articles",
    #     type=Path,
    #     default=DEFAULT_ARTICLES,
    #     help="Path to article JSONL file.",
    # )
    parser.add_argument(
        "--qa",
        type=Path,
        default=DEFAULT_QA,
        help="Path to QA JSONL file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for EDA outputs.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    # article_path = args.articles.resolve()
    qa_path = args.qa.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # article_records = read_jsonl(article_path)
    qa_records = read_jsonl(qa_path)

    # article_df, category_df, article_summary = summarize_articles(article_records)
    qa_df, qa_type_df, qa_summary = summarize_qa(qa_records)

    # article_df.to_csv(
    #     out_dir / "article_eda_rows.csv",
    #     index=False,
    #     encoding="utf-8-sig",
    # )
    # category_df.to_csv(
    #     out_dir / "article_category_distribution.csv",
    #     index=False,
    #     encoding="utf-8-sig",
    # )

    qa_df.to_csv(
        out_dir / "qa_eda_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    qa_type_df.to_csv(
        out_dir / "qa_type_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    chart_paths = save_charts(
        qa_df,
        qa_type_df,
        out_dir,
    )

    # summary = {"articles": article_summary, "qa": qa_summary}
    summary = {"qa": qa_summary}

    (out_dir / "eda_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (out_dir / "eda_report.md").write_text(
        make_markdown_report(
            qa_summary,
            qa_type_df,
        ),
        encoding="utf-8",
    )

    print_summary(qa_summary)

    print()
    print(f"Saved EDA outputs to: {out_dir}")

    if chart_paths:
        print(f"Saved charts to: {out_dir / 'charts'}")
    else:
        print("Charts were not generated because matplotlib is not installed.")


if __name__ == "__main__":
    main()