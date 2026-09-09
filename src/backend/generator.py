"""
Module 4: Generation Stage (MOCK)
Phụ trách: Người 4 (My)
Mô tả: Đây là file giả lập để Tuấn Anh (Integration Owner) test luồng pipeline.
Khi nào My code xong sẽ ghi đè logic thật vào file này.
"""

from typing import Callable, List
from src.RAG.retrieval.schema import EvidencePlan, CoverageMatrix, RouteDecision, GenerationDecision

def run_generation_stage(
    question: str,
    evidence_plan: EvidencePlan,
    coverage_matrix: List[CoverageMatrix],
    route_decision: RouteDecision,
    ranked_candidates: List[dict],
    retry_retrieval: Callable | None = None,
) -> GenerationDecision:
    """
    Hàm giả lập (Mock) quá trình gọi LLM để sinh câu trả lời.
    """
    print("\n[MOCK GENERATOR] Đang nhận dữ liệu từ Router...")
    print(f" - Câu hỏi: {question}")
    print(f" - Số lượng Candidates nhận được: {len(ranked_candidates)}")
    
    # Trả về một quyết định giả lập
    return GenerationDecision(
        decision="ANSWER",
        answer="Đây là câu trả lời giả lập (Mock) từ Generator. My sẽ thay thế bằng code gọi LLM thật sau.",
        citations=["Nguồn 1", "Nguồn 2"],
        refusal_reason="",
        missing_evidence=[]
    )