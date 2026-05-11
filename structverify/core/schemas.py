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


class ValueRole(str, Enum):
    """
    Claim의 value가 어떤 의미적 역할인지 — KOSIS와의 직접 비교 가능 여부 결정.

    [v6.2 김예슬 - 2026-05-08]
    "14도를 넘겼다"의 14, "1.5도 마지노선"의 1.5는 측정값이 아니라 기준선/임계값.
    이런 값을 KOSIS와 직접 비교하면 가짜 mismatch가 발생함.
    """
    MEASUREMENT = "measurement"  # 직접 측정값 (14.8도, 5천만명) → KOSIS 비교
    THRESHOLD   = "threshold"    # 기준선/임계값 ("14도를 넘겼다", "1.5도 이상") → 비교 제외
    DELTA       = "delta"        # 변화량/차이 ("2.3도 웃돌았다", "0.18도 더 높다") → 별도 검증 필요
    RANK        = "rank"         # 순위 ("4위", "역대 1위") → 별도 검증 필요
    RATIO       = "ratio"        # 비율/배수 ("2.6배", "30% 증가") → 별도 검증 필요
    NONE        = "none"         # 검증 부적합


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
    # [v6.2 김예슬] value의 역할 분류 — KOSIS 직접 비교 가능 여부 결정
    # measurement만 verify_claim()이 KOSIS row와 직접 비교
    # threshold/rank/none은 즉시 unverifiable, delta/ratio는 별도 검증 (TODO)
    value_role: ValueRole = ValueRole.MEASUREMENT
    # [v6.3] evidence_plan — 검증에 필요한 시점들의 명세
    # measurement: 1개, delta/ratio: 2개, rank/none: 0개
    evidence_plan: "EvidencePlan | None" = None
    comparison_type: ClaimType | None = None
    source_reference: str | None = None
    graph_schema_candidates: list[dict[str, str]] = Field(default_factory=list)



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

class EvidenceRequirement(BaseModel):
    """
    [v6.3] EvidencePlan의 단일 항목.
    delta/ratio 검증을 위한 endpoint, 또는 measurement 검증의 primary 등.
    """
    role: str  # "primary" | "endpoint_a" | "endpoint_b" | "comparison"
    label: str | None = None  # "current", "baseline", "1991-04", 등 사람이 읽는 레이블
    indicator: str | None = None
    time_period: str | None = None
    population: str | None = None


class EvidencePlan(BaseModel):
    """
    [v6.3] claim 검증에 필요한 evidence들의 명세서.
    schema_inductor가 LLM으로 생성. value_role에 따라 형태 결정:
      - measurement: requirements=[primary] (1개)
      - delta:      requirements=[endpoint_a, endpoint_b] (2개)
      - ratio:      requirements=[endpoint_a, endpoint_b] (2개)
      - rank/none:  requirements=[] (KOSIS 직접 비교 불가)

    combiner는 verifier가 value_role 기반으로 자동 선택.
    LLM이 임의 수식을 생성하면 안전성 문제 → enum으로 제한.
    """
    requirements: list[EvidenceRequirement] = Field(default_factory=list)
    combiner: str = "direct"  # "direct" | "delta" | "ratio_pct"


class Evidence(BaseModel):
    source_name: str
    stat_table_id: str | None = None
    official_value: float | None = None
    unit: str | None = None
    time_period: str | None = None
    raw_response: dict[str, Any] = Field(default_factory=dict)
    graph_nodes: list[GraphNode] = Field(default_factory=list)
    provenance: ProvenanceRecord | None = None
    # [v6.3] 어떤 EvidenceRequirement에 매핑되는지
    requirement_role: str | None = None
    requirement_label: str | None = None


class VerificationResult(BaseModel):
    result_id: UUID = Field(default_factory=uuid4)
    claim_id: UUID
    verdict: VerdictType
    confidence: float = 0.0
    evidence: Evidence | None = None  # 주(primary) evidence (하위 호환)
    # [v6.3] delta/ratio 검증을 위한 보조 evidences
    supplementary_evidences: list[Evidence] = Field(default_factory=list)
    # [v6.3] combiner가 계산한 값 (검증에 실제로 사용된 값)
    computed_value: float | None = None
    combiner_used: str | None = None
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


# [v6.3] Forward reference 해결 — ClaimSchema가 EvidencePlan을 참조
ClaimSchema.model_rebuild()