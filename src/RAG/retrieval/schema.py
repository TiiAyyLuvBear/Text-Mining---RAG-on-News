from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def result_from_chunk(row: dict[str, object], score: float) -> dict[str, object]:
    metadata = dict(row.get("metadata") or {})
    return {
        "chunk_id": str(row["chunk_id"]),
        "article_id": str(row["article_id"]),
        "score": float(score),
        "text": str(row.get("text") or ""),
        "chunk_text": str(row.get("chunk_text") or ""),
        "title": metadata.get("title"),
        "category": metadata.get("category"),
        "chunk_index": metadata.get("chunk_index"),
    }

# 2. QUERY PLANNER SCHEMAS
class SubQuestion(BaseModel):
    id: str
    text: str
    evidence_type: Literal[
        "FACT", "TEMPORAL_FACT", "RELATION", "CAUSAL", "LIST",
        "COMPARATIVE_CONCLUSION",
    ]
    required_concepts: List[str] = Field(default_factory=list)
    required_entities: List[str] = Field(default_factory=list)
    required_numbers: List[str] = Field(default_factory=list)
    required_dates: List[str] = Field(default_factory=list)
    temporal_constraints: List[str] = Field(default_factory=list)
    answer_operator: Optional[str] = None


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


# DOWNSTREAM CONTRACTS (Dùng chung cho cả nhóm)
class Candidate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    chunk_id: str
    article_id: str
    text: str
    title: Optional[str] = ""
    retrieval_score: float = Field(default=0.0, alias="score")
    rerank_score: Optional[float] = None
    rank: Optional[int] = None
    citation_rank: Optional[int] = None
    applied_boosts: List[str] = Field(default_factory=list)


class CandidateSupport(BaseModel):
    chunk_id: str
    article_id: str
    support_score: float = 0.0
    supports: bool = False
    lexical_score: float = 0.0
    entity_score: float = 1.0
    concept_score: float = 1.0
    temporal_score: float = 1.0
    relation_score: float = 1.0
    failure_reason: str = ""
    matched_unit: str = ""


class CoverageMatrix(BaseModel):
    sub_question_id: str
    candidates: List[CandidateSupport] = Field(default_factory=list)
    covered: bool = False
    covered_by_articles: List[str] = Field(default_factory=list)
    missing_sub_questions: List[str] = Field(default_factory=list)
    best_candidate: Optional[CandidateSupport] = None
    failure_reason: str = ""


class RouteDecision(BaseModel):
    route: Literal["SINGLE_DOC", "REQUIRES_MULTI_DOC", "INSUFFICIENT"]
    reason: str
    covered_sub_questions: List[str] = Field(default_factory=list)
    missing_sub_questions: List[str] = Field(default_factory=list)
    selected_article_ids: List[str] = Field(default_factory=list)
    retry_allowed: bool = True


class Citation(BaseModel):
    citation_rank: int
    article_id: Optional[str] = None
    chunk_id: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None


class GenerationDecision(BaseModel):
    decision: Literal["ANSWER", "REFUSE"]
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    refusal_reason: str = ""
    refusal_reason_code: str = ""
    failure_category: str = ""
    missing_evidence: List[str] = Field(default_factory=list)
    verification_status: Literal["PASS", "WARNING", "FAIL", "NOT_RUN"] = "NOT_RUN"
    verification_errors: List[dict[str, Any]] = Field(default_factory=list)
    verification_warnings: List[dict[str, Any]] = Field(default_factory=list)
    compression_stats: dict[str, Any] = Field(default_factory=dict)
    retry_count: int = 0
    retry_query: str = ""
