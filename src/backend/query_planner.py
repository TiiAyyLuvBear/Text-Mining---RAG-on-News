"""Regex/heuristic query analysis that produces the shared EvidencePlan."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from src.RAG.retrieval.schema import EvidencePlan, SubQuestion
from src.backend.temporal_retrieval import extract_temporal_terms

_QUESTION_NOISE = {
    "ai",
    "bao",
    "diễn",
    "khi",
    "ngày",
    "năm",
    "những",
    "sau",
    "số",
    "so",
    "tại",
    "tháng",
    "theo",
    "thông",
    "trong",
    "trước",
    "vai",
    "vì",
}
_NUMBER_PATTERN = re.compile(
    r"(?<![\w/.-])\d+(?:[.,]\d+)*(?:\s*(?:%|nghìn|ngàn|triệu|tỷ))?(?![\w/.-])",
    re.IGNORECASE,
)


def normalize_question(question: Any) -> str:
    """Normalize Unicode and whitespace without throwing on non-string input."""
    return re.sub(
        r"\s+", " ", unicodedata.normalize("NFC", str(question or ""))
    ).strip()


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _extract_entities(question: str) -> list[str]:
    matches = [
        match
        for match in re.finditer(r"\b[\wÀ-ỹ.-]+\b", question, re.UNICODE)
        if any(char.isalpha() for char in match.group(0))
        and match.group(0)[0].isupper()
        and match.group(0).casefold() not in _QUESTION_NOISE
    ]
    entities: list[str] = []
    index = 0
    while index < len(matches):
        parts = [matches[index].group(0)]
        end = matches[index].end()
        cursor = index + 1
        while cursor < len(matches) and question[end : matches[cursor].start()].isspace():
            parts.append(matches[cursor].group(0))
            end = matches[cursor].end()
            cursor += 1
        entities.append(" ".join(parts))
        index = cursor
    return _unique(entities)


def extract_entities_numbers_dates(question: str) -> dict[str, list[str]]:
    """Extract stable heuristic features; temporal parsing has one owner."""
    temporal_constraints = extract_temporal_terms(question)
    dates = [
        term
        for term in temporal_constraints
        if re.search(r"\d|tháng|năm|ngày", term, re.IGNORECASE)
    ]

    temporal_spans = [
        match.span()
        for term in temporal_constraints
        for match in re.finditer(re.escape(term), question, re.IGNORECASE)
    ]
    numbers = [
        match.group(0)
        for match in _NUMBER_PATTERN.finditer(question)
        if not any(start <= match.start() and match.end() <= end for start, end in temporal_spans)
    ]

    return {
        "dates": _unique(dates),
        "numbers": _unique(numbers),
        "entities": _extract_entities(question),
        "temporal_constraints": temporal_constraints,
    }


def classify_answer_operator(
    question: str,
    extracted_features: dict[str, list[str]] | None = None,
) -> str:
    """Classify the answer operation; comparison takes precedence."""
    del extracted_features
    q_lower = question.casefold()
    if any(
        word in q_lower
        for word in (
            "so sánh",
            "khác nhau",
            "giống nhau",
            "so với",
            "hơn kém",
            "khác biệt",
            "điểm chung",
        )
    ):
        return "COMPARE"
    if any(
        word in q_lower
        for word in (
            "khi nào",
            "năm nào",
            "bao giờ",
            "diễn biến",
            "lịch sử",
            "thời gian",
            "trình tự",
            "sắp xếp theo thời gian",
            "qua các thời kỳ",
            "tiến trình",
        )
    ):
        return "TIMELINE"
    if any(
        word in q_lower
        for word in (
            "tại sao",
            "vì sao",
            "do đâu",
            "nguyên nhân",
            "lý do",
            "hậu quả",
            "hệ quả",
            "điều gì khiến",
            "tóm tắt",
            "nội dung chính",
        )
    ):
        return "CAUSAL_SUMMARY"
    return "DIRECT"


def analyze_intent(question: str) -> dict[str, Any]:
    """Compatibility API for explicit-source and claim hints from the upstream branch."""
    q_lower = question.casefold()
    return {
        "is_multi_doc": any(
            phrase in q_lower
            for phrase in (
                "ba bài báo",
                "các bài báo",
                "cả ba bài",
                "hai bài báo",
                "từ các nguồn",
            )
        ),
        "is_claim": "đúng hay sai" in q_lower or "nhận định" in q_lower,
        "operator": classify_answer_operator(question),
    }


def _comparison_focus(question: str, left: str, right: str, focus: str) -> str:
    base = re.sub(
        r"^\s*(?:hãy\s+)?so\s+sánh\s+",
        "",
        question,
        flags=re.IGNORECASE,
    )
    pair = re.compile(
        rf"\b{re.escape(left)}\s+(?:và|so với)\s+{re.escape(right)}\b",
        re.IGNORECASE,
    )
    focused, count = pair.subn(focus, base, count=1)
    return focused if count else f"{base} (đối tượng: {focus})"


def _between_comparison_subjects(question: str) -> tuple[str, str, str] | None:
    """Extract ``metric between left and right`` comparisons before NER hints.

    Vietnamese organisation names commonly contain lowercase words (for
    example ``Trường ĐH Công nghệ Giao thông vận tải``), so capitalization-only
    entity extraction is not a safe way to split this construction.
    """
    match = re.search(
        r"(?is)^(.*?)\bgiữa\s+(.+?)\s+và\s+(.+?)(?=[.?!](?:\s|$)|$)",
        question,
    )
    if not match:
        return None
    metric = re.sub(r"^\s*(?:hãy\s+)?so\s+sánh\s+", "", match.group(1), flags=re.IGNORECASE)
    metric = metric.strip(" ,:;-\t")
    left = match.group(2).strip(" ,:;-\t")
    right = match.group(3).strip(" ,:;-\t")
    if not metric or not left or not right:
        return None
    return metric, left, right


def _explicit_article_topics(question: str) -> list[str]:
    """Return explicitly enumerated ``bài về ...`` evidence topics."""
    return _unique([
        match.group(1).strip()
        for match in re.finditer(r"(?i)\bbài\s+về\s+([^,.;?]+)", question)
    ])


def _requires_direct_comparative_conclusion(question: str) -> bool:
    """Detect subjective rankings that source facts alone cannot establish."""
    value = question.casefold()
    return bool(
        re.search(r"\b(?:nghiêm trọng|quan trọng|đáng kể|tốt|xấu)\s+nhất\b", value)
        or re.search(r"\byếu tố nào\b.+\b(?:hơn|nhất)\b", value)
        or re.search(r"\b(?:xếp hạng|được.+cho là)\b", value)
    )


def build_sub_questions(
    question: str,
    operator: str | dict[str, Any],
    features: dict[str, list[str]] | None = None,
) -> list[SubQuestion]:
    """Split comparisons without discarding the requested metric or period."""
    intent = operator if isinstance(operator, dict) else {}
    operator = str(intent.get("operator") or operator)
    if intent.get("is_claim"):
        claim_text = re.sub(
            r"(?i)(?:nhận định sau(?: đây)? )?đúng hay sai\s*:\s*",
            "",
            question,
        ).strip(" \"'")
        return [
            SubQuestion(
                id="sq1",
                text=f"Kiểm chứng thông tin: {claim_text}",
                evidence_type="FACT",
            )
        ]
    if operator == "COMPARE":
        topics = _explicit_article_topics(question)
        if len(topics) >= 2:
            sub_questions = [
                SubQuestion(
                    id=f"sq{index}",
                    text=topic,
                    evidence_type="RELATION",
                )
                for index, topic in enumerate(topics, start=1)
            ]
            if _requires_direct_comparative_conclusion(question):
                sub_questions.append(SubQuestion(
                    id=f"sq{len(sub_questions) + 1}",
                    text="So sánh trực tiếp hoặc xếp hạng: " + "; ".join(topics),
                    evidence_type="COMPARATIVE_CONCLUSION",
                    required_concepts=topics,
                ))
            return sub_questions
        between = _between_comparison_subjects(question)
        if between:
            metric, left, right = between
            return [
                SubQuestion(id="sq1", text=f"{metric} {left}", evidence_type="RELATION"),
                SubQuestion(id="sq2", text=f"{metric} {right}", evidence_type="RELATION"),
            ]
        entities = (features or {}).get("entities", [])
        if len(entities) >= 2:
            left, right = entities[:2]
            return [
                SubQuestion(
                    id="sq1",
                    text=_comparison_focus(question, left, right, left),
                    evidence_type="RELATION",
                ),
                SubQuestion(
                    id="sq2",
                    text=_comparison_focus(question, left, right, right),
                    evidence_type="RELATION",
                ),
            ]
        return [SubQuestion(id="sq1", text=question, evidence_type="RELATION")]
    evidence_type = {
        "TIMELINE": "TEMPORAL_FACT",
        "CAUSAL_SUMMARY": "CAUSAL",
    }.get(operator, "FACT")
    return [SubQuestion(id="sq1", text=question, evidence_type=evidence_type)]


def build_evidence_plan(question: Any) -> EvidencePlan:
    """Build a deterministic shared EvidencePlan, including a safe empty fallback."""
    normalized = normalize_question(question)
    features = extract_entities_numbers_dates(normalized)
    intent = analyze_intent(normalized)
    operator = intent["operator"]
    hint_map = {
        "COMPARE": "COMPARISON",
        "TIMELINE": "TIMELINE",
        "CAUSAL_SUMMARY": "GENERAL",
        "DIRECT": "FACTOID",
    }
    return EvidencePlan(
        normalized_question=normalized,
        query_type_hint=hint_map[operator],
        entities=features["entities"],
        numbers=features["numbers"],
        dates=features["dates"],
        temporal_constraints=features["temporal_constraints"],
        estimated_sources_needed=(
            3
            if intent["is_multi_doc"]
            else 2
            if operator in {"COMPARE", "TIMELINE"}
            else 1
        ),
        answer_operator=operator,
        sub_questions=build_sub_questions(normalized, intent, features),
    )
