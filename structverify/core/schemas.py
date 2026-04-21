"""
core/schemas.py — v2.0 전체 파이프라인 데이터 모델

모든 계층 간 데이터 교환은 이 스키마를 통해 이루어진다.
v2.0에서 Graph Anchor, SIR Tree, Provenance, Feedback 스키마 추가.

[참고] IBM Docling (2024) — https://github.com/DS4SD/docling
  SIR Tree의 block 구조 + entity_refs/graph_anchor_ids 설계에 참고
[참고] FEVER (Thorne et al., NAACL 2018)
  3단계 판정 체계 (SUPPORTS/REFUTES/NEI) → match/mismatch/unverifiable
[참고] AutoSchemaKG (arXiv 2505.23628) — https://github.com/NousResearch/AutoSchemaKG
  LLM이 도메인 스키마를 자동 유도하는 Schema Induction 구조 참고
  
[DONE] 김예슬
- 팀원 코드가 붙기 전에 입출력 계약부터 고정
- SIR Tree 구조 + Claim/Graph/Evidence/VerificationResult 모델 설계
- 검증 결과에 mismatch_type, explanation, provenance_summary 필드 추가
- GraphNode/GraphEdge 모델 추가 (v2)
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


# ── Enums ──

class SourceType(str, Enum):
    URL = "url"; PDF = "pdf"; DOCX = "docx"; TEXT = "text"

class BlockType(str, Enum):
    PARAGRAPH = "paragraph"; TABLE = "table"; HEADING = "heading"; LIST = "list"

class ClaimType(str, Enum):
    """주장 유형 — ClaimBuster (Hassan et al., VLDB 2017) 분류 체계 확장"""
    INCREASE = "increase"; DECREASE = "decrease"; SCALE = "scale"
    COMPARISON = "comparison"; FORECAST = "forecast"

class VerdictType(str, Enum):
    """판정 결과 — FEVER 3단계 매핑"""
    MATCH = "match"; MISMATCH = "mismatch"; UNVERIFIABLE = "unverifiable"

class MismatchType(str, Enum):
    VALUE = "value"; TIME_PERIOD = "time_period"
    POPULATION = "population"; EXAGGERATION = "exaggeration"

class FeedbackType(str, Enum):
    HUMAN_REVIEW = "human_review"; LOW_CONFIDENCE = "low_confidence"
    FAILURE = "failure"; DRIFT = "drift"

class GraphNodeType(str, Enum):
    CLAIM = "claim"; ENTITY = "entity"; METRIC = "metric"
    TIME = "time"; EVIDENCE = "evidence"; SOURCE = "source"

class GraphEdgeType(str, Enum):
    MEASURED_AT = "measured_at"; BELONGS_TO = "belongs_to"
    VERIFIED_BY = "verified_by"; SOURCED_FROM = "sourced_from"
    CONTRADICTS = "contradicts"; SUPPORTS = "supports"


# ── SIR Tree (with Graph Anchors) ──

class SourceOffset(BaseModel):
    """원본 문서 내 위치 — 원문 역추적용"""
    page: int | None = None
    char_start: int = 0
    char_end: int = 0

class Sentence(BaseModel):
    """개별 문장 + 그래프 앵커"""
    sent_id: str
    text: str
    char_offset_start: int = 0
    char_offset_end: int = 0
    has_numeric: bool = False
    graph_anchor_id: str | None = None  # v2: Graph 노드 연결용

class SIRBlock(BaseModel):
    """
    SIR Tree의 기본 블록 — Docling DocItem 확장
    v2: entity_refs, event_refs, graph_anchor_ids 추가
    """
    block_id: str
    type: BlockType
    level: int | None = None
    content: str | None = None
    sentences: list[Sentence] = []
    headers: list[str] | None = None
    rows: list[list[str]] | None = None
    entity_refs: list[str] = []           # v2: 엔티티 참조
    event_refs: list[str] = []            # v2: 이벤트 참조
    graph_anchor_ids: list[str] = []      # v2: 그래프 노드 연결
    source_offset: SourceOffset = SourceOffset()

class SIRDocument(BaseModel):
    """SIR Tree 문서 전체"""
    doc_id: UUID = Field(default_factory=uuid4)
    source_type: SourceType
    source_uri: str | None = None
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    blocks: list[SIRBlock] = []
    detected_domain: str | None = None    # v2: 도메인 판별 결과


# ── Claim ──

class ClaimSchema(BaseModel):
    """
    주장 핵심 정보 — ProgramFC 분해 구조 + AutoSchemaKG 동적 유도
    v2: graph_schema_candidates 추가
    """
    indicator: str | None = None
    time_period: str | None = None
    unit: str | None = None
    population: str | None = None
    value: float | None = None
    comparison_type: ClaimType | None = None
    source_reference: str | None = None
    graph_schema_candidates: list[dict[str, str]] = []  # v2: LLM이 유도한 그래프 스키마 후보

class Claim(BaseModel):
    claim_id: UUID = Field(default_factory=uuid4)
    doc_id: UUID
    block_id: str
    sent_id: str
    claim_text: str
    claim_type: ClaimType | None = None
    schema: ClaimSchema | None = None
    source_offset: SourceOffset = SourceOffset()
    check_worthy_score: float = 0.0
    graph_anchor_id: str | None = None    # v2: 그래프 노드 ID


# ── Graph ──

class GraphNode(BaseModel):
    """그래프 노드"""
    node_id: str
    node_type: GraphNodeType
    label: str
    domain: str | None = None
    properties: dict[str, Any] = {}

class GraphEdge(BaseModel):
    """그래프 엣지"""
    edge_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    from_node: str
    to_node: str
    edge_type: GraphEdgeType
    weight: float = 1.0
    properties: dict[str, Any] = {}

class ProvenanceRecord(BaseModel):
    """데이터 출처 이력"""
    provenance_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    source_connector: str
    source_id: str | None = None
    query_used: str | None = None
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    raw_snapshot: dict[str, Any] = {}


# ── Evidence / Verification ──

class Evidence(BaseModel):
    source_name: str
    stat_table_id: str | None = None
    official_value: float | None = None
    unit: str | None = None
    time_period: str | None = None
    raw_response: dict[str, Any] = {}
    graph_nodes: list[GraphNode] = []     # v2: Evidence 그래프 노드
    provenance: ProvenanceRecord | None = None  # v2: 출처 이력

class VerificationResult(BaseModel):
    result_id: UUID = Field(default_factory=uuid4)
    claim_id: UUID
    verdict: VerdictType
    confidence: float = 0.0
    evidence: Evidence | None = None
    mismatch_type: MismatchType | None = None
    explanation: str | None = None
    provenance_summary: str | None = None  # v2: 출처 경로 요약
    reviewer_verdict: VerdictType | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Feedback / Adaptation ──

class FeedbackEvent(BaseModel):
    """Human Review 및 실패 사례 기록"""
    event_id: UUID = Field(default_factory=uuid4)
    claim_id: UUID
    feedback_type: FeedbackType
    original_verdict: VerdictType | None = None
    corrected_verdict: VerdictType | None = None
    reviewer_note: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class DomainPack(BaseModel):
    """도메인 팩 메타데이터"""
    pack_id: str
    domain: str
    version: str
    config: dict[str, Any] = {}
    adapter_path: str | None = None
    eval_score: float | None = None
    is_active: bool = True


# ── Pipeline Report ──

class VerificationReport(BaseModel):
    report_id: UUID = Field(default_factory=uuid4)
    document: SIRDocument
    claims: list[Claim] = []
    results: list[VerificationResult] = []
    graph_nodes: list[GraphNode] = []     # v2
    graph_edges: list[GraphEdge] = []     # v2
    feedbacks: list[FeedbackEvent] = []   # v2
    domain_pack_used: str | None = None   # v2
    created_at: datetime = Field(default_factory=datetime.utcnow)
    summary: str | None = None
