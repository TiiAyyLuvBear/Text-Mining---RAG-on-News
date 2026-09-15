"""Regex/heuristic query analysis that produces the shared EvidencePlan."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from src.RAG.retrieval.schema import EvidencePlan, SubQuestion
from src.backend.temporal_retrieval import extract_temporal_terms

# 1. EXPANDED NOISE SET: Added sentence starters and common leading Vietnamese nouns
_QUESTION_NOISE = {
    # Pronouns, demonstratives, question starters
    "ai", "bao", "nào", "này", "đó", "đây", "kia", "ấy", "gì", "những", "các", "mọi", "mỗi", "một",
    "liệu", "nếu", "sao", "thế", "đâu", "chăng",
    # Common leading nouns, measurement words, sentence starters
    "số", "mức", "hạng", "loại", "sự", "việc", "ngày", "tháng", "năm", "lần", "khi", "lúc", "thời", "điều",
    "phần", "mục", "lý", "nguyên", "hệ", "hậu", "kết", "điểm", "cuộc", "đợt", "giấy", "hoa", "khu", "tình",
    "xét", "chợ", "tòa", "chương", "chuyến", "thực", "quan", "nhân", "cậu", "bộ", "hàm", "vụ", "bão",
    "ông", "bà", "anh", "chị", "hộ", "thí", "phòng", "luật", "cơ", "chi", "show", "đội",
    # Prepositions, conjunctions
    "của", "cho", "do", "với", "về", "vì", "tại", "trong", "trên", "dưới", "ngoài", "từ", "tới", "đến", "ở",
    "và", "hay", "hoặc", "nhưng", "mà", "thì", "là", "bằng", "như", "theo", "dựa", "qua",
    # Common verbs & modifiers
    "có", "không", "làm", "để", "thấy", "xem", "xin", "so", "khác", "giống", "diễn", "tóm", "tiến", "trình",
    "đang", "đã", "sẽ", "sắp", "được", "bị", "hãy", "nên", "cần", "phải", "kiểm", "đúng",
    # Other words
    "quá", "còn", "cùng", "cách", "cái", "hơn", "nhất", "chỉ", "cũng", "vẫn", "cứ", "chưa", "đều", "thông",
    "bệnh", "hãng", "công", "ty", "tập", "đoàn", "trường", "viện", "nhà", "nước", "nhóm", "bên", "ngành", "lĩnh", "vực", "dự", "án", "vai", "sau", "mạnh", "quế"
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
    """Extract capitalized entity candidates, filtering noise words and joining codes."""
    tokens = list(re.finditer(r"(?<!\w)[\wÀ-ỹ.'’-]+(?!\w)", question, re.UNICODE))
    entities: list[str] = []
    index = 0

    while index < len(tokens):
        token = tokens[index]
        word = token.group(0)

        if (
            any(char.isalpha() for char in word)
            and word[0].isupper()
            and word.casefold() not in _QUESTION_NOISE
        ):
            parts = [word]
            end = token.end()
            cursor = index + 1

            while cursor < len(tokens):
                next_token = tokens[cursor]
                next_word = next_token.group(0)

                if not question[end : next_token.start()].isspace() and end != next_token.start():
                    break

                is_upper = any(c.isalpha() for c in next_word) and next_word[0].isupper()
                is_number_or_code = next_word.isdigit() or (any(c.isdigit() for c in next_word) and next_word.isupper())

                if (is_upper and next_word.casefold() not in _QUESTION_NOISE) or is_number_or_code:
                    parts.append(next_word)
                    end = next_token.end()
                    cursor += 1
                else:
                    break

            entity = " ".join(parts).strip(" ,:;.-")
            if entity and entity.casefold() not in _QUESTION_NOISE:
                entities.append(entity)
            index = cursor
        else:
            index += 1

    return _unique(entities)


def extract_entities_numbers_dates(question: str) -> dict[str, list[str]]:
    """Extract stable heuristic features; temporal parsing has one owner."""
    temporal_constraints = extract_temporal_terms(question)

    clean_dates = []
    for term in temporal_constraints:
        if re.search(r"\d|tháng|năm|ngày", term, re.IGNORECASE):
            cleaned = re.sub(r"^(?:trong|vào|từ|đến|ở)\s+", "", term, flags=re.IGNORECASE).strip()
            clean_dates.append(cleaned)

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
        "dates": _unique(clean_dates),
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

    if re.search(r"\bđược\s+so\s+sánh\b.*\b(?:nào|gì)\b", q_lower):
        return "DIRECT"

    if (
        re.search(r"\btừ\s+(?:năm|tháng|ngày)\b.+\bđến\b", q_lower)
        or re.search(r"\bqua\s+(?:ba|các|\d+)\s+(?:thời điểm|bài báo)\b", q_lower)
        or "theo thời gian" in q_lower
    ):
        return "TIMELINE"

    if any(
        word in q_lower
        for word in (
            "so sánh", "khác nhau", "giống nhau", "so với",
            "hơn kém", "khác biệt", "điểm chung",
        )
    ):
        return "COMPARE"
    if any(
        word in q_lower
        for word in (
            "khi nào", "năm nào", "bao giờ", "diễn biến",
            "lịch sử", "thời gian", "trình tự", "sắp xếp theo thời gian",
            "qua các thời kỳ", "tiến trình",
        )
    ):
        return "TIMELINE"
    if any(
        word in q_lower
        for word in (
            "tại sao", "vì sao", "do đâu", "nguyên nhân",
            "lý do", "hậu quả", "hệ quả", "điều gì khiến",
            "tóm tắt", "nội dung chính",
        )
    ):
        return "CAUSAL_SUMMARY"
    return "DIRECT"


def analyze_intent(question: str) -> dict[str, Any]:
    """Compatibility API for explicit-source and claim hints."""
    q_lower = question.casefold()
    return {
        "is_multi_doc": any(
            phrase in q_lower
            for phrase in (
                "ba bài báo", "các bài báo", "cả ba bài",
                "hai bài báo", "từ các nguồn",
            )
        ),
        "is_claim": "đúng hay sai" in q_lower or "nhận định" in q_lower,
        "operator": classify_answer_operator(question),
    }


def _comparison_focus(question: str, left: str, right: str, focus: str) -> str:
    """Isolate focus entity cleanly without destructive sub-string deletions."""
    base = re.sub(
        r"^\s*(?:hãy\s+)?so\s+sánh\s+",
        "",
        question,
        flags=re.IGNORECASE,
    ).strip(" ,:;-\t")

    pair = re.compile(
        rf"\b{re.escape(left)}\b\s+(?:và|so với)\s+\b{re.escape(right)}\b",
        re.IGNORECASE,
    )
    focused, count = pair.subn(focus, base, count=1)
    if count:
        return re.sub(r"\s+", " ", focused).strip()

    # SAFE FOCUS: Do NOT use regex to delete the 'other' entity to prevent
    # breaking compound words (e.g., 'Ngữ văn' -> 'văn', 'Trung ương' -> 'ương')
    return f"Đối với {focus}: {base}"


def _between_comparison_subjects(question: str) -> tuple[str, str, str] | None:
    """Extract comparison targets across varied Vietnamese comparative structures."""
    # Pattern 1: So với A... của/thì B...
    m_so_voi = re.search(
        r"(?is)^so\s+với\s+([^,.]+?)\s*,\s*(.+?)\s+của\s+([^,.]+?)(?=[.?!]|[\r\n]|$)", 
        question
    )
    if m_so_voi:
        return m_so_voi.group(2).strip(), m_so_voi.group(1).strip(), m_so_voi.group(3).strip()

    # Pattern 2: Đối với A và B có những điểm khác biệt gì
    m_doi_voi = re.search(
        r"(?is)(?:đối\s+với)\s+([^,.]+?)\s+và\s+([^,.]+?)(?:\s+nói\s+chung)?\s+(?:có\s+(?:những\s+)?(?:điểm\s+)?khác\s+biệt)",
        question
    )
    if m_doi_voi:
        return "Khác biệt", m_doi_voi.group(1).strip(), m_doi_voi.group(2).strip()

    # Pattern 3: Explicit article topics matching 'bài về ...' anywhere in the clause
    bai_matches = list(re.finditer(r"(?i)\bbài\s+về\s+([^,.;?]+?)(?=\s+và\b|\s*:\s*|[,.;?]|$)", question))
    if len(bai_matches) >= 2:
        return "Nội dung", bai_matches[0].group(1).strip(), bai_matches[1].group(1).strip()

    # Pattern 4: Explicit giữa/của A và B (prevent early cutoff at abbreviations like TP.HCM)
    match = re.search(
        r"(?is)^(.*?)\b(?:giữa|của)\s+([^,]+?)\s+và\s+([^,]+?)(?=[.?!](?:\s|$)|$)",
        question,
    )
    if match:
        metric = re.sub(r"^\s*(?:hãy\s+)?so\s+sánh\s+", "", match.group(1), flags=re.IGNORECASE).strip(" ,:;-\t")
        left = match.group(2).strip(" ,:;-\t")
        right = match.group(3).strip(" ,:;-\t")
        if not metric:
            metric = "Thông tin"
        if left and right:
            return metric, left, right

    # Pattern 5: A... khác biệt/khác nhau ra sao so với B (support dots in abbreviations)
    m_so_sanh_vs = re.search(
        r"(?is)^(.*?)\s+(?:khác biệt|khác nhau)\s+.*?\s+so\s+với\s+([^?!]+?)(?=[.?!](?:\s|$)|[\r\n]|$)", 
        question
    )
    if m_so_sanh_vs:
        left = m_so_sanh_vs.group(1).strip(" ,:;-\t")
        right = m_so_sanh_vs.group(2).strip(" ,:;-\t")
        if left and right:
            return "Khác biệt", left, right

    return None


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


def _explicit_time_points(question: str) -> list[str]:
    """Find specific date/time anchors for timeline queries."""
    pattern = re.compile(
        r"(?i)(?<![\w/.-])(?:"
        r"\d{1,2}[./]\d{1,2}[./]\d{4}|"
        r"(?:ngày\s+\d+(?:\.\d+)?(?:\s*\(\d{1,2}\.\d{1,2}\))?)|"
        r"(?:tháng\s+\d{1,2}/\d{4})|"
        r"(?:tháng\s+\d{1,2})|"
        r"năm\s+(?:19|20)\d{2}|"
        r"(?:19|20)\d{2}"
        r")(?![\w/.-])"
    )
    return _unique([match.group(0) for match in pattern.finditer(question)])


def _timeline_sub_questions(question: str) -> list[SubQuestion] | None:
    """Anchors sub-questions at explicit time points without corrupting the question text."""
    points = _explicit_time_points(question)
    if len(points) < 2:
        return None

    # Keep original question body intact so context is not destroyed
    return [
        SubQuestion(
            id=f"sq{index}",
            text=f"Tại thời điểm {point}: {question}",
            evidence_type="TEMPORAL_FACT",
        )
        for index, point in enumerate(points, start=1)
    ]


def build_sub_questions(
    question: str,
    operator: str | dict[str, Any],
    features: dict[str, list[str]] | None = None,
) -> list[SubQuestion]:
    """Split comparisons and timelines deterministically."""
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

    time_points = _explicit_time_points(question)
    if (operator == "TIMELINE" or (operator == "COMPARE" and len(time_points) >= 3)) and len(time_points) >= 2:
        timeline = _timeline_sub_questions(question)
        if timeline:
            return timeline

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

            has_left_time = bool(re.search(r"\b(?:\d{4}|tháng|năm|ngày)\b", left, re.IGNORECASE))
            shared_time = ""
            if not has_left_time:
                for term in (features or {}).get("temporal_constraints", []):
                    clean_term = term.strip()
                    pattern = rf"(?i)\s*(?:trong|vào|từ|đến|ở)?\s*{re.escape(clean_term)}\b"
                    match_term = re.search(pattern, right)
                    if match_term:
                        shared_time = match_term.group(0)
                        right = right[:match_term.start()] + right[match_term.end():]
                        right = right.strip(" ,:;-\t")
                        break

            sub_1_text = re.sub(r"\s+", " ", f"{metric} {left}{shared_time}").strip()
            sub_2_text = re.sub(r"\s+", " ", f"{metric} {right}{shared_time}".strip()).strip()

            return [
                SubQuestion(id="sq1", text=sub_1_text, evidence_type="RELATION"),
                SubQuestion(id="sq2", text=sub_2_text, evidence_type="RELATION"),
            ]

        entities = (features or {}).get("entities", [])
        if len(entities) >= 2:
            left, right = entities[0], entities[1]
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

    sub_questions = build_sub_questions(normalized, intent, features)

    for sub_question in sub_questions:
        requirements = extract_entities_numbers_dates(sub_question.text)
        sub_question.required_entities = requirements["entities"]
        sub_question.required_numbers = requirements["numbers"]
        sub_question.required_dates = requirements["dates"]
        sub_question.temporal_constraints = requirements["temporal_constraints"]
        sub_question.answer_operator = operator

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
        sub_questions=sub_questions,
    )