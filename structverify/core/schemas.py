"""
core/schemas.py — v2.1 전체 파이프라인 데이터 모델

v2.1 변경점
- Sentence에 regex 중심 has_numeric 대신
  has_numeric_surface + candidate_score/candidate_label 추가
- claim candidate detection을 독립 태스크로 다루기 위한 필드 추가
- 기존 has_numeric는 하위 호환을 위해 제거하지 않고 property처럼 대체 가능하게 설계

설계 의도
- surface rule은 보조 신호로만 사용
- 실제 검증 후보 여부는 candidate_score / candidate_label이 담당
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────

class SourceType(str, Enum):
    URL = "url"
    PDF = "pdf"
    DOCX = "docx"
    TEXT = "text"


class BlockType(str, Enum):
    PARAGRAPH = "paragraph"
    TABLE = "table"
    HEADING = "heading"
    LIST = "list"


class ClaimType(str, Enum):
    """주장 유형 — ClaimBuster 계열 분류 확장"""
    INCREASE = "increase"
    DECREASE = "decrease"
    SCALE = "scale"
    COMPARISON = "comparison"
    FORECAST = "forecast"


class VerdictType(str, Enum):
    """판정 결과 — FEVER 3단계 매핑"""
    MATCH = "match"
    MISMATCH = "mismatch"
    UNVERIFIABLE = "unverifiable"


class MismatchType(str, Enum):
    VALUE = "value"
    TIME_PERIOD = "time_period"
    POPULATION = "population"
    EXAGGERATION = "exaggeration"


class FeedbackType(str, Enum):
    HUMAN_REVIEW = "human_review"
    LOW_CONFIDENCE = "low_confidence"
    FAILURE = "failure"
    DRIFT = "drift"


class GraphNodeType(str, Enum):
    CLAIM = "claim"
    ENTITY = "entity"
    METRIC = "metric"
    TIME = "time"
    EVIDENCE = "evidence"
    SOURCE = "source"


class GraphEdgeType(str, Enum):
    MEASURED_AT = "measured_at"
    BELONGS_TO = "belongs_to"
    VERIFIED_BY = "verified_by"
    SOURCED_FROM = "sourced_from"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"


# ── SIR Tree ─────────────────────────────────────────────────

class SourceOffset(BaseModel):
    """원문 역추적용 절대 위치 정보"""
    page: int | None = None
    char_start: int = 0
    char_end: int = 0


class Sentence(BaseModel):
    """
    개별 문장 + candidate detection 결과

    필드 설명
    - has_numeric_surface:
        정규식 기반의 약한 표면 신호. 최종 candidate 판단이 아님.
    - candidate_score:
        검증 후보 점수. 0~1 범위.
    - candidate_label:
        threshold를 적용한 최종 후보 여부.
    - candidate_source:
        점수의 출처 ("surface_rule", "weak_supervision", "teacher_llm" 등)
    - candidate_signals:
        디버깅/분석용 보조 신호
    """
    sent_id: str
    text: str
    char_offset_start: int = 0
    char_offset_end: int = 0

    # 기존 regex 탐지는 하위 호환을 위해 surface signal로 격하
    has_numeric_surface: bool = False

    # 논문형 candidate detection 결과
    candidate_score: float = 0.0
    candidate_label: bool = False
    candidate_source: str | None = None
    candidate_signals: dict[str, Any] = Field(default_factory=dict)

    graph_anchor_id: str | None = None

    @property
    def has_numeric(self) -> bool:
        """
        하위 호환용 property.
        기존 코드가 s.has_numeric를 참조하더라도 surface signal로 동작하게 한다.
        """
        return self.has_numeric_surface


class SIRBlock(BaseModel):
    block_id: str
    type: BlockType
    level: int | None = None
    content: str | None = None
    sentences: list[Sentence] = Field(default_factory=list)
    headers: list[str] | None = None
    rows: list[list[str]] | None = None
    entity_refs: list[str] = Field(default_factory=list)
    event_refs: list[str] = Field(default_factory=list)
    graph_anchor_ids: list[str] = Field(default_factory=list)
    source_offset: SourceOffset = Field(default_factory=SourceOffset)


class SIRDocument(BaseModel):
    doc_id: UUID = Field(default_factory=uuid4)
    source_type: SourceType
    source_uri: str | None = None
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    blocks: list[SIRBlock] = Field(default_factory=list)
    detected_domain: str | None = None


# ── Claim ────────────────────────────────────────────────────

class ClaimSchema(BaseModel):
    indicator: str | None = None
    time_period: str | None = None
    unit: str | None = None
    population: str | None = None
    value: float | None = None
    comparison_type: ClaimType | None = None
    source_reference: str | None = None
    graph_schema_candidates: list[dict[str, str]] = Field(default_factory=list)


class Claim(BaseModel):
    claim_id: UUID = Field(default_factory=uuid4)
    doc_id: UUID
    block_id: str
    sent_id: str
    claim_text: str
    claim_type: ClaimType | None = None
    schema: ClaimSchema | None = None
    source_offset: SourceOffset = Field(default_factory=SourceOffset)
    check_worthy_score: float = 0.0
    graph_anchor_id: str | None = None


# ── Graph ────────────────────────────────────────────────────

class GraphNode(BaseModel):
    node_id: str
    node_type: GraphNodeType
    label: str
    domain: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    edge_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    from_node: str
    to_node: str
    edge_type: GraphEdgeType
    weight: float = 1.0
    properties: dict[str, Any] = Field(default_factory=dict)


class ProvenanceRecord(BaseModel):
    provenance_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    source_connector: str
    source_id: str | None = None
    query_used: str | None = None
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    raw_snapshot: dict[str, Any] = Field(default_factory=dict)


# ── Evidence / Verification ──────────────────────────────────

class Evidence(BaseModel):
    source_name: str
    stat_table_id: str | None = None
    official_value: float | None = None
    unit: str | None = None
    time_period: str | None = None
    raw_response: dict[str, Any] = Field(default_factory=dict)
    graph_nodes: list[GraphNode] = Field(default_factory=list)
    provenance: ProvenanceRecord | None = None


class VerificationResult(BaseModel):
    result_id: UUID = Field(default_factory=uuid4)
    claim_id: UUID
    verdict: VerdictType
    confidence: float = 0.0
    evidence: Evidence | None = None
    mismatch_type: MismatchType | None = None
    explanation: str | None = None
    provenance_summary: str | None = None
    reviewer_verdict: VerdictType | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Feedback / Adaptation ────────────────────────────────────

class FeedbackEvent(BaseModel):
    event_id: UUID = Field(default_factory=uuid4)
    claim_id: UUID
    feedback_type: FeedbackType
    original_verdict: VerdictType | None = None
    corrected_verdict: VerdictType | None = None
    reviewer_note: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DomainPack(BaseModel):
    pack_id: str
    domain: str
    version: str
    config: dict[str, Any] = Field(default_factory=dict)
    adapter_path: str | None = None
    eval_score: float | None = None
    is_active: bool = True


# ── Report ───────────────────────────────────────────────────

class VerificationReport(BaseModel):
    report_id: UUID = Field(default_factory=uuid4)
    document: SIRDocument
    claims: list[Claim] = Field(default_factory=list)
    results: list[VerificationResult] = Field(default_factory=list)
    graph_nodes: list[GraphNode] = Field(default_factory=list)
    graph_edges: list[GraphEdge] = Field(default_factory=list)
    feedbacks: list[FeedbackEvent] = Field(default_factory=list)
    domain_pack_used: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    summary: str | None = None