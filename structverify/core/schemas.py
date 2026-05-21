# [2026-05-14 | 이수민] memory/v1: working memory 도메인 가드용 필드 추가
#   - Evidence.category_path: KOSIS catalog의 카테고리 경로 (예: "인구 > 출생")
#   - MismatchType.DOMAIN_MISMATCH: stat_id 카테고리가 문서 도메인과 어긋난 경우
"""
core/schemas.py — v3 전체 파이프라인 데이터 모델

v2 변경점
- Sentence에 regex 중심 has_numeric 대신
  has_numeric_surface + candidate_score/candidate_label 추가
- claim candidate detection을 독립 태스크로 다루기 위한 필드 추가
- 기존 has_numeric는 하위 호환을 위해 제거하지 않고 property처럼 대체 가능하게 설계

설계 의도
- surface rule은 보조 신호로만 사용
- 실제 검증 후보 여부는 candidate_score / candidate_label이 담당

v3 변경점 [김예슬]
- GraphEdgeType에 4개 추가
- COMPARE 엣지: indicator가 같은 Claim 쌍을 전부 연결
"쉬었음인구"를 indicator로 가진 C1과 C2가 COMPARE로 연결되면,
"2.6배" 검증 시 MetricNode 2-hop으로 C1+C2를 함께 KOSIS에 조회해서 비율 계산이 가능
- 문맥 엣지: sir_doc이 넘어오면 extract_context_edges()를 호출해서 NEXT_SENT/IN_BLOCK/IN_DOC를 GraphEdge로 변환해 반환값에 포함

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
    # [이수민 2026-05-14] working memory 도메인 가드용
    DOMAIN_MISMATCH = "domain_mismatch"  # stat_id 카테고리가 문서 도메인과 어긋남


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
    # ── 멀티홉 시간 그래프용 (document_graph.py) ──────────────────────────
    DOCUMENT = "document"            # 문서 메타 (anchor_year property 보유)
    SENTENCE = "sentence"            # 문장 (REFERS_TO 타겟, sir_doc 호환)
    TEMPORAL_EXPR = "temporal_expr"  # "작년", "9월", "재작년 같은 기간"
    RESOLVED_TIME = "resolved_time"  # 절대 시점 (2023, 2024-09 등)


class GraphEdgeType(str, Enum):
    MEASURED_AT = "measured_at"
    BELONGS_TO = "belongs_to"
    VERIFIED_BY = "verified_by"
    SOURCED_FROM = "sourced_from"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    # GraphRAG 문맥 엣지 (sir_builder.extract_context_edges → graph_builder)
    NEXT_SENT = "next_sent"    # 문장 → 다음 문장 (문맥 흐름)
    IN_BLOCK  = "in_block"     # 문장 → 소속 문단
    IN_DOC    = "in_doc"       # 문단 → 소속 문서
    # 복합 주장 검증용 (같은 지표를 공유하는 Claim 간)
    COMPARE   = "compare"      # C1 ↔ C2 (2.6배 같은 파생 주장 검증)
    # ── 멀티홉 시간 그래프용 (document_graph.py) ──────────────────────────
    HAS_TEMPORAL = "has_temporal"  # Sentence/Claim → TemporalExpr
    RELATIVE_TO  = "relative_to"   # TemporalExpr → Document (anchor 의존)
    RESOLVES_TO  = "resolves_to"   # TemporalExpr → ResolvedTime
    REFERS_TO    = "refers_to"     # TemporalExpr → 다른 Sentence (coref)


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
    parent_path: str | None         # "노동 > 청년 > 쉬었음 인구" (계층 카테고리)
    is_approximate: bool = False    # "안팎" 인지
    modifier: str | None            # 근사 표현 원문
    prev_value: float | None = None
    prev_time_period: str | None = None
    prev_phrase: str | None = None

    # [2026-05-21] schema_inductor가 한 claim → N sub-claim 분기 시,
    # 각 sub-claim의 *검증 역할*을 명시. planner LLM이 같은 claim_text를 보고
    # 둘 다 동일 plan_type으로 잘못 분류하는 걸 방지하는 1차 신호.
    #   - "base"             : 단일 값 단순 확인 (예: 출생아 수 = 20717명)
    #                          → plan_type=absolute, 시퀀스: catalog → fetch → finish
    #   - "derived_rate"     : 비율/증가율/감소율 계산 (단위 %, ~증가율 류)
    #                          → plan_type=growth_rate
    #   - "derived_difference": 절대 차이/변화량 (단위 절대단위, 차이값)
    #                          → plan_type=difference
    #   - None               : 명시 안 됨 — planner LLM이 claim_text/schema로 추론
    value_role: str | None = None



class Claim(BaseModel):
    claim_id: UUID = Field(default_factory=uuid4)
    doc_id: UUID
    block_id: str
    sent_id: str
    claim_text: str
    claim_type: str | None = None          # 자유 문자열
    canonical_type: ClaimType | None = None
    schema: ClaimSchema | None = None
    source_offset: SourceOffset = Field(default_factory=SourceOffset)
    check_worthy_score: float = 0.0
    graph_anchor_id: str | None = None
    # [v4 김예슬] 앞뒤 문맥 (SIR Tree에서 추출, runtime_agent가 부착)
    # "이는 2.6배" 같은 대명사 참조 해소 + schema_inductor/query_builder에서 활용
    context_text: str | None = None


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
    # [이수민 2026-05-14] working memory 도메인 가드용
    category_path: str | None = None  # KOSIS catalog의 "인구 > 출생" 등


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
    # ★ Phase E: growth_rate/diff 같은 계산 검증값
    computed_value: float | None = None
    """Calculate tool 결과값 — 주장값 vs 이 값 비교로 verdict 결정."""
    formula: str | None = None
    """사용된 수식 (예: '(current - prev) / prev * 100')."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    # [Multi-hop GraphRAG - 김예슬 2026-05-08]
    # 파생 주장(2.6배 등)을 COMPARE 엣지 2-hop으로 검증한 경우 표시
    multihop_used: bool = False
    multihop_detail: dict | None = None


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