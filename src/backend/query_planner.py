"""
Module 1: Query Analysis & Evidence Planner
Objective: Parse raw questions into an EvidencePlan using Regex and Heuristics, 
strictly without using LLMs to optimize latency.
"""

import re
import unicodedata
from typing import List

# Import directly from the team's shared schema
from src.RAG.retrieval.schema import SubQuestion, EvidencePlan

# ==========================================
# 1. PREPROCESSING & ENTITY EXTRACTION
# ==========================================

def normalize_question(question: str) -> str:
    """Normalize Unicode and remove extra whitespace."""
    if not question:
        return ""
    q = unicodedata.normalize('NFC', str(question))
    return re.sub(r'\s+', ' ', q).strip()

def extract_entities_numbers_dates(question: str) -> dict:
    """Quickly extract entities, numbers, and dates using Regex."""
    # Extract time phrases: year 19xx or 20xx (with or without the word "năm"), month 1-12, day 1-31
    dates = re.findall(
        r'(?:\bnăm\s+)?\b(?:19|20)\d{2}\b|\btháng\s+\d{1,2}\b|\bngày\s+\d{1,2}\b', 
        question, re.IGNORECASE
    )
    # Extract numbers (integer or decimal)
    raw_numbers = re.findall(r'\b\d+(?:[.,]\d+)?\b', question)
    
    # Exclude numbers that are part of date/year phrases (extract all numbers present in dates)
    date_numbers = set(re.findall(r'\b\d+\b', " ".join(dates)))
    numbers = [num for num in raw_numbers if num not in date_numbers]
    
    # Quickly extract capitalized entities (e.g., Vingroup, Hà Nội)
    entities = re.findall(r'([A-ZĐ][a-zà-ỹ]+(?:\s+[A-ZĐ][a-zà-ỹ]+)*)', question)
    
    return {
        "dates": list(set(dates)),
        "numbers": list(set(numbers)),
        "entities": list(set(entities))
    }

# ==========================================
# 2. INTENT DETECTION
# ==========================================

def analyze_intent(question: str) -> dict:
    q_lower = question.lower()
    
    # 1. Multi-doc check
    multi_doc_kws = ["ba bài báo", "các bài báo", "cả ba bài", "hai bài báo", "từ các nguồn"]
    is_multi_doc = any(kw in q_lower for kw in multi_doc_kws)
    
    # 2. Claim check
    is_claim = "đúng hay sai" in q_lower or "nhận định" in q_lower
    
    # 3. Phân loại Operator
    operator = "DIRECT"
    
    if any(w in q_lower for w in ["so sánh", "khác nhau", "giống nhau", "so với", "hơn kém", "khác biệt", "điểm chung"]):
        operator = "COMPARE"
        
    # Chỉ coi là TIMELINE nếu thực sự yêu cầu chuỗi/diễn biến/tiến trình
    elif any(w in q_lower for w in ["diễn biến", "lịch sử", "trình tự", "sắp xếp theo thời gian", "qua các thời kỳ", "tiến trình"]):
        operator = "TIMELINE"
        
    elif any(w in q_lower for w in [
        "tại sao", "vì sao", "nguyên nhân", "lý do", "hậu quả", "hệ quả", "điều gì khiến",
        "tóm tắt", "nội dung chính", "diễn ra như thế nào",
        "mục đích", "ý nghĩa", "vai trò", "nhiệm vụ"
    ]):
        operator = "CAUSAL_SUMMARY"
        
    # Lưu ý: "khi nào", "năm nào", "ngày nào", "bao giờ" sẽ rơi vào DIRECT -> Factoid -> 1 nguồn

    return {
        "is_multi_doc": is_multi_doc,
        "is_claim": is_claim,
        "operator": operator
    }

def build_sub_questions(question: str, intent: dict) -> List[SubQuestion]:
    """Split complex questions into sub-questions based on Intent."""
    operator = intent["operator"]
    
    # Priority 1: Handle Claim Verification questions
    if intent["is_claim"]:
        # Extract the content of the claim to be verified
        claim_text = re.sub(r'(?i)(đúng hay sai\s*:\s*|nhận định sau đúng hay sai\s*:\s*|nhận định sau đây đúng hay sai\s*:\s*)', '', question).strip(' "\'')
        return [SubQuestion(id="sq1", text=f"Kiểm chứng thông tin: {claim_text}", evidence_type="FACT")]
    
    # Priority 2: Handle Comparison questions (Split the clause in half if possible)
    if operator == "COMPARE":
        # Remove multi-doc noise words to split the sentence more accurately
        clean_q = re.sub(r'(?i)(trong ba bài báo|của ba bài báo|theo các bài báo|dựa trên thông tin)', '', question)
        for kw in [" và ", " so với "]:
            if kw in clean_q:
                parts = clean_q.split(kw, 1)
                # Extract adjacent entities around conjunctions to create scoping sub-questions
                sub1 = " ".join(parts[0].split()[-3:])
                sub2 = " ".join(parts[1].split()[:3])
                return [
                    SubQuestion(id="sq1", text=f"Tìm thông tin về {sub1}?", evidence_type="RELATION"),
                    SubQuestion(id="sq2", text=f"Tìm thông tin về {sub2}?", evidence_type="RELATION"),
                    SubQuestion(id="sq3", text=question, evidence_type="RELATION") # Keep the original question for synthesis
                ]
        return [SubQuestion(id="sq1", text=question, evidence_type="RELATION")]
        
    # Remaining cases
    elif operator == "TIMELINE":
        return [SubQuestion(id="sq1", text=question, evidence_type="TEMPORAL_FACT")]
    elif operator == "CAUSAL_SUMMARY":
        return [SubQuestion(id="sq1", text=question, evidence_type="CAUSAL")]
    else:
        return [SubQuestion(id="sq1", text=question, evidence_type="FACT")]

# ==========================================
# 3. MAIN ENTRY POINT
# ==========================================

def build_evidence_plan(question: str) -> EvidencePlan:
    """Encapsulate the logic to generate the retrieval plan."""
    try:
        norm_q = normalize_question(question)
        if not norm_q:
            raise ValueError("Câu hỏi rỗng.")
            
        features = extract_entities_numbers_dates(norm_q)
        intent = analyze_intent(norm_q)
        operator = intent["operator"]
        
        # Map operator to query_type_hint
        hint_map = {
            "COMPARE": "COMPARISON",
            "TIMELINE": "TIMELINE",
            "CAUSAL_SUMMARY": "GENERAL",
            "DIRECT": "FACTOID"
        }
        
        # Dynamic determination of the number of sources (Sources Needed)
        if intent["is_multi_doc"]:
            est_sources = 3
        elif operator == "COMPARE":
            est_sources = 2
        elif operator == "TIMELINE":
            # Nếu câu hỏi thực sự là dòng thời gian phức tạp mới cần 2 nguồn
            est_sources = 2
        else:
            est_sources = 1
            
        sub_questions = build_sub_questions(norm_q, intent)
        
        return EvidencePlan(
            normalized_question=norm_q,
            query_type_hint=hint_map[operator],
            entities=features["entities"],
            numbers=features["numbers"],
            dates=features["dates"],
            temporal_constraints=features["dates"],
            estimated_sources_needed=est_sources,
            answer_operator=operator,
            sub_questions=sub_questions
        )
    except Exception as e:
        print(f"[QueryPlanner] Warning: Fallback used for '{question}'. Error: {e}")
        return EvidencePlan(
            normalized_question=question, 
            query_type_hint="GENERAL", 
            answer_operator="DIRECT",
            sub_questions=[SubQuestion(id="sq1", text=question, evidence_type="FACT")]
        )

# ==========================================
# 4. RUN EXPERIMENTAL TESTS
# ==========================================
if __name__ == "__main__":
    # Real test queries sampled from Dataset/data_QA_Convert.jsonl
    test_queries = [
        # 1. Factoid (mốc thời gian cụ thể -> Factoid, 1 nguồn)
        "Vụ tai nạn trên cao tốc TP.HCM - Trung Lương xảy ra vào thời gian nào?",
        # 2. Cause-Effect (hỏi lý do -> CAUSAL_SUMMARY, 1 nguồn)
        "Vì sao Bí thư Tỉnh ủy Khánh Hòa lo ngại người dân có thể chủ quan trước bão Kalmaegi?",
        # 3. Single-doc Comparison (so sánh đơn văn bản -> COMPARE, 2 nguồn)
        "Hàm lượng purin trong cá nhỏ và nội tạng động vật khác biệt ra sao so với thịt đỏ thông thường?",
        # 4. Claim Verification (xác minh nhận định -> DIRECT với sub-query bóc tách, 1 nguồn)
        "Đúng hay sai: 'Chỉ cần bỏ thịt dê và chuyển sang ăn hải sản là có thể kiểm soát tốt axit uric'?",
        # 5. Multi-doc Comparison (so sánh đa văn bản -> COMPARE, 3 nguồn)
        "So sánh các bệnh nền (bệnh lý đi kèm) được ghi nhận ở những bệnh nhân nặng mắc cúm A và sốt xuất huyết đang điều trị tại Bệnh viện Bệnh nhiệt đới Trung ương trong các bài báo.",
        # 6. Multi-doc Timeline (dòng thời gian đa bài báo -> TIMELINE, 3 nguồn)
        "Trình tự thời gian các sự kiện liên quan đến án phạt của FIFA, kháng cáo lên CAS, và kế hoạch nhập tịch lần 2 của Malaysia diễn ra như thế nào?"
    ]

    for idx, q in enumerate(test_queries, 1):
        print(f"\n[{idx}] Question: {q}")
        plan = build_evidence_plan(q)
        print(f"    -> Query Hint    : {plan.query_type_hint}")
        print(f"    -> Answer Op     : {plan.answer_operator}")
        print(f"    -> Sources Needed: {plan.estimated_sources_needed}")
        print(f"    -> Sub-queries   : {[sq.text for sq in plan.sub_questions]}")
        print(f"    -> Dates         : {plan.dates}")