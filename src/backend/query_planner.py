"""
Module 1: Query Analysis & Evidence Planner
Phụ trách: Người 1 (Tuấn Anh)
Mục tiêu: Phân tích câu hỏi thô thành EvidencePlan bằng Regex và Heuristic, 
tuyệt đối không dùng LLM để tối ưu latency.
"""

import re
import unicodedata
from typing import List, Literal
from pydantic import BaseModel, Field
from src.RAG.retrieval.schema import SubQuestion, EvidencePlan

# ==========================================
# 1. DATA CONTRACTS (Pydantic Models)
# ==========================================

class SubQuestion(BaseModel):
    id: str
    text: str
    evidence_type: Literal["FACT", "TEMPORAL_FACT", "RELATION", "CAUSAL", "LIST"]

class EvidencePlan(BaseModel):
    normalized_question: str
    query_type_hint: Literal["FACTOID", "COMPARISON", "TIMELINE", "GENERAL"]
    entities: List[str] = Field(default_factory=list)
    numbers: List[str] = Field(default_factory=list)
    dates: List[str] = Field(default_factory=list)
    temporal_constraints: List[str] = Field(default_factory=list)
    estimated_sources_needed: int = 1
    answer_operator: Literal["DIRECT", "COMPARE", "TIMELINE", "CAUSAL_SUMMARY"]
    sub_questions: List[SubQuestion] = Field(default_factory=list)

# ==========================================
# 2. XỬ LÝ LÕI (Core Logic)
# ==========================================

def normalize_question(question: str) -> str:
    """Chuẩn hóa Unicode và loại bỏ khoảng trắng thừa."""
    if not question:
        return ""
    q = unicodedata.normalize('NFC', str(question))
    q = re.sub(r'\s+', ' ', q).strip()
    return q

def extract_entities_numbers_dates(question: str) -> dict:
    """Trích xuất nhanh các thực thể, số liệu và thời gian bằng Regex."""
    # Bắt cụm thời gian: năm 19xx hoặc 20xx (có hoặc không có từ "năm"), tháng 1-12, ngày 1-31
    dates = re.findall(
        r'(?:\bnăm\s+)?\b(?:19|20)\d{2}\b|\btháng\s+\d{1,2}\b|\bngày\s+\d{1,2}\b', 
        question, 
        re.IGNORECASE
    )
    # Bắt các con số (nguyên hoặc thập phân)
    raw_numbers = re.findall(r'\b\d+(?:[.,]\d+)?\b', question)
    # Loại trừ các con số đã thuộc cụm ngày/năm (trích xuất toàn bộ số có trong dates)
    date_numbers = set(re.findall(r'\b\d+\b', " ".join(dates)))
    numbers = [num for num in raw_numbers if num not in date_numbers]
    # Bắt nhanh thực thể viết hoa (ví dụ: Vingroup, Hà Nội)
    entities = re.findall(r'([A-ZĐ][a-zà-ỹ]+(?:\s+[A-ZĐ][a-zà-ỹ]+)*)', question)
    
    return {
        "dates": list(set(dates)),
        "numbers": list(set(numbers)),
        "entities": list(set(entities))
    }

def classify_answer_operator(question: str) -> str:
    """Phân loại toán tử truy vấn dựa trên từ khóa heuristic."""
    q_lower = question.lower()
    
    if any(w in q_lower for w in ["so sánh", "khác nhau", "giống nhau", "so với", "hơn kém"]):
        return "COMPARE"
    if any(w in q_lower for w in ["khi nào", "năm nào", "bao giờ", "diễn biến", "lịch sử", "thời gian"]):
        return "TIMELINE"
    if any(w in q_lower for w in ["tại sao", "nguyên nhân", "lý do", "hậu quả", "tóm tắt"]):
        return "CAUSAL_SUMMARY"
        
    return "DIRECT"

def build_sub_questions(question: str, operator: str) -> List[SubQuestion]:
    """Tách câu hỏi phức tạp thành các sub-questions đơn giản."""
    if operator == "COMPARE":
        # Thử bẻ câu bằng liên từ phổ biến
        split_keywords = [" và ", " so với "]
        for kw in split_keywords:
            if kw in question:
                parts = question.split(kw, 1)
                return [
                    SubQuestion(
                        id="sq1", 
                        text=f"Thông tin chi tiết về {parts[0].split()[-1]} là gì?", 
                        evidence_type="RELATION"
                    ),
                    SubQuestion(
                        id="sq2", 
                        text=f"Thông tin chi tiết về {parts[1].split()[0]} là gì?", 
                        evidence_type="RELATION"
                    )
                ]
        # Fallback nếu không chẻ được câu
        return [SubQuestion(id="sq1", text=question, evidence_type="RELATION")]
        
    elif operator == "TIMELINE":
        return [SubQuestion(id="sq1", text=question, evidence_type="TEMPORAL_FACT")]
        
    elif operator == "CAUSAL_SUMMARY":
        return [SubQuestion(id="sq1", text=question, evidence_type="CAUSAL")]
        
    else:
        return [SubQuestion(id="sq1", text=question, evidence_type="FACT")]

# ==========================================
# 3. HÀM GIAO TIẾP CHÍNH (Main Entry Point)
# ==========================================

def build_evidence_plan(question: str) -> EvidencePlan:
    """Đóng gói toàn bộ logic để trả về EvidencePlan hoàn chỉnh."""
    try:
        norm_q = normalize_question(question)
        if not norm_q:
            raise ValueError("Câu hỏi rỗng.")
            
        features = extract_entities_numbers_dates(norm_q)
        operator = classify_answer_operator(norm_q)
        sub_questions = build_sub_questions(norm_q, operator)
        
        # Ánh xạ operator sang query_type_hint
        hint_map = {
            "COMPARE": "COMPARISON",
            "TIMELINE": "TIMELINE",
            "CAUSAL_SUMMARY": "GENERAL",
            "DIRECT": "FACTOID"
        }
        
        # Nếu là COMPARE thì khả năng cao cần >= 2 nguồn
        est_sources = 2 if operator == "COMPARE" else 1
        
        return EvidencePlan(
            normalized_question=norm_q,
            query_type_hint=hint_map[operator],
            entities=features["entities"],
            numbers=features["numbers"],
            dates=features["dates"],
            temporal_constraints=features["dates"], # Gắn tạm dates làm constraints
            estimated_sources_needed=est_sources,
            answer_operator=operator,
            sub_questions=sub_questions
        )
    except Exception as e:
        # Fallback an toàn nếu có lỗi bất ngờ (tránh crash toàn hệ thống)
        print(f"[QueryPlanner] Warning: Fallback used for question '{question}'. Error: {e}")
        return EvidencePlan(
            normalized_question=question, 
            query_type_hint="GENERAL", 
            answer_operator="DIRECT",
            sub_questions=[SubQuestion(id="sq1", text=question, evidence_type="FACT")]
        )

# ==========================================
# 4. CHẠY TEST THỬ NGHIỆM
# ==========================================
if __name__ == "__main__":
    test_queries = [
        "Vingroup được thành lập vào năm nào?",
        "So sánh doanh thu của FPT và Hòa Phát trong năm 2023.",
        "Tại sao thị trường bất động sản đóng băng?",
        "        Ai là người sáng lập ra tập đoàn Viettel   ???"
    ]
    
    for q in test_queries:
        print(f"\n--- Câu hỏi gốc: {q}")
        plan = build_evidence_plan(q)
        print(plan.model_dump_json(indent=2))