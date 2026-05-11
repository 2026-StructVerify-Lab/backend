"""
graph/document_graph.py — 문서 레벨 시간 그래프 빌더 (멀티홉 핵심)

[설계 의도]
LLM agent가 문서 전체를 한 번에 보고 다음 정보를 추출하여 그래프 노드/엣지로 박는다:
  - 문서의 anchor 시점 (이 문서가 기준으로 삼는 연도/시점)
  - 모든 시간 표현 (어떤 언어든, 어떤 형태든)
  - 시간 표현 간 참조 관계 (coreference)

[멀티홉 검증 흐름]
ClaimNode "작년 평균기온 14.8도"
  ─ HAS_TEMPORAL → TemporalExprNode("작년", sent=s0002)
                       ├─ RELATIVE_TO → DocumentNode(anchor_year=2024)
                       └─ RESOLVES_TO → ResolvedTimeNode("2023")
                                              ↑
schema_inductor가 ClaimGraph로 traverse하여 resolved time을 prompt에 주입
verifier가 KOSIS row 매칭 시 resolved time을 사용

[도메인 확장성]
- LLM agent가 모든 해석을 담당. 룰 매핑 일체 없음.
- 한국어 "작년/재작년/지난해", 영어 "last year/yesterday", 중국어 "去年" 모두 처리.
- 새 도메인 추가 시 prompt 수정 없이 작동 (LLM 일반화).

[참고]
- HOVER (Jiang et al., EMNLP 2020): 멀티홉 사실 검증
- GraphRAG (arXiv 2501.00309): 그래프 기반 다중 hop 추론
"""
from __future__ import annotations

import copy
from typing import Any

from structverify.core.schemas import (
    GraphEdge, GraphEdgeType, GraphNode, GraphNodeType, SIRDocument,
)
from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


# ── LLM agent prompt ────────────────────────────────────────────────────────
# 핵심: 룰 매핑 없음, LLM이 anchor 추출 + 모든 시간 표현 + coref를 동시에 풀어냄

_TEMPORAL_AGENT_PROMPT = """당신은 문서의 시간 정보를 정밀하게 분석하는 전문가입니다.
문서 전체를 읽고 두 가지 정보를 JSON으로 추출하세요.

# 추출 대상

## 1) anchor: 이 문서가 "기준 시점"으로 삼는 시점

[v6.7 우선순위 — 위에서부터 엄격히 적용]

★★★ 1순위 (절대 우선): 본문 어디든 (시작/중간/끝) 명시된 **발행일/입력일/작성일** 연도
   발행일 패턴 — 다음 중 *어떤 것이라도* 발견되면 그 연도가 anchor_year:
   · "입력 YYYY.MM.DD" / "입력 YYYY-MM-DD"  (예: "입력 2025.01.01 14:59")
   · "발행 YYYY.MM.DD" / "게재 YYYY.MM.DD"
   · "작성일 YYYY..." / "보도 YYYY..."
   · 문서 위/아래에 떨어진 "YYYY-MM-DD" 또는 "YYYY.MM.DD" 단독 라인
   · "YYYY년 MM월 DD일자" 형식
   · 기자 이름과 함께 나오는 날짜
   · 본문 맨 첫 줄/두 번째 줄에 있어도, **맨 끝에 있어도** 동일하게 우선.

   ⚠️ 발행일은 본문 *어디든* 있을 수 있습니다. 위치 무관. 찾으면 그게 anchor.

★★ 2순위: 발행일이 없을 때만 — "OOOO년은", "OOOO년 들어" 같은 본문 명시 표현
   · 단, *발행일 패턴이 있으면 이 신호는 무시*.
   · 예: 발행일="2025.01.01"이고 본문에 "2024년은 ~한 해"가 있어도 anchor=2025.
     ("2024년은 ~한 해"는 anchor가 아니라 *2024년에 대한 서술*임)

★ 3순위: "올해", "현재" 같은 표현의 참조 시점 추론
   · 다른 단서 없으면 추론하지 말고 null

본문에 명시된 단서가 없으면 anchor_year=null (추측 금지).

⚠️ 매우 흔한 실수 1: 발행일이 "2025-06-25"인데 본문에서 "지난해 같은 달" 같은 표현을 보고
   "지난해=2024"이니까 anchor도 2024라고 박는 실수. anchor는 *기준* 시점입니다.
   "지난해"는 anchor에서 1 뺀 거고, anchor 자체는 발행일 연도(2025)입니다.

⚠️ 매우 흔한 실수 2: 발행일이 "2025.01.01"인데 본문에 "2024년은 ~한 해"라는 강한 표현이
   있어서 anchor=2024로 박는 실수. **발행일이 더 강한 신호입니다.**
   "2024년은 ~한 해"는 2024년에 대한 *서술*이지 anchor가 아닙니다.
   올바른 답: anchor_year=2025, "작년"=2024, "재작년"=2023.

[구체 예시 — 그대로 따라하세요]
예시 A:
  본문: "# 작년 연평균기온 14.8도\n입력 2025.01.01 14:59\n2024년은 ... 뜨거웠던 해\n작년 평균기온은 14.8도..."
  → 발행일 패턴 "입력 2025.01.01" 발견 → anchor_year=2025
  → "작년" → 2024 (anchor - 1)
  → "재작년" → 2023 (anchor - 2)
  → "2024년"은 본문에 있어도 *절대 표현*이므로 그대로 2024 (anchor 결정에 사용 안 함)

예시 B:
  본문: "올해 4월 출생아 수 ... 6.7% 늘었다\n... [중간 생략] ...\n(2025-06-25 작성)"
  → 본문 *맨 끝*에 "2025-06-25" → 발행일 패턴 → anchor_year=2025
  → "올해" → 2025, "지난해" → 2024

예시 C (발행일 없는 경우):
  본문: "2024년은 압도적으로 뜨거웠던 해로 남았다. 작년 평균기온은 ..."
  → 발행일 패턴 없음 → 2순위: "2024년은 ~ 해" → anchor_year=2024
  → "작년" → 2023

anchor_evidence에 어떤 문장/표현을 근거로 정했는지 짧게 적으세요.
(예: "본문 2번째 줄 '입력 2025.01.01'", "본문 끝 '(2025-06-25 작성)'")

[참고 — v6.5 이전 가이드 (위 우선순위가 적용 안 되면)]
- 본문에 "OOOO년은", "지난 OOOO년", "발행일: OOOO" 등으로 명시된 연도/시점을 우선
- 명시된 anchor가 여러 개라면 본문 서술 시점의 기준이 되는 것을 선택
- 본문에 명시되지 않은 경우 anchor_year=null
- 무엇을 근거로 anchor를 정했는지 anchor_evidence에 짧게 기록

## 2) temporal_expressions: 본문의 모든 시간 표현 목록
모든 시간 표현을 빠짐없이 수집하고, anchor와 다른 문장 참조를 활용해 절대 시점으로 풀어내세요.

시간 표현의 예시 (어떤 언어든):
- 상대 표현: "작년", "재작년", "지난해", "올해", "내년", "전년", "last year", "去年"
- 부분 시점: "9월", "1∼11월", "Q3", "지난 분기", "여름철"
- 참조 표현: "이는", "같은 기간", "the same period", "それ"
- 절대 표현: "2024년", "2024-09", "1973년"

각 표현마다:
- sent_id: 그 표현이 등장하는 문장 id
  ⚠️ **반드시 입력에서 [...] 안에 표시된 정확한 ID 문자열을 그대로 복사**할 것.
  예: 입력에 "[b0001_s0003] ..."이면 sent_id="b0001_s0003" 으로 답하세요.
  prefix를 생략하거나 줄여서 "s0003"으로 답하지 마세요.
- expression: 원문 그대로의 표현
- resolved: 절대 시점 문자열로 풀어냄
  · 단일 연도: "2023"
  · 연-월: "2024-09"
  · 기간 범위: "2022-01..2022-11"
  · 풀어낼 수 없으면 null
- resolution_basis: 풀이 근거를 한 줄로
  · 예: "anchor_year - 1 = 2023"
  · 예: "s0009의 '1∼11월'을 anchor_year-2(2022)에 적용"
- refers_to_sent_id: "이는", "같은 기간" 등 다른 문장의 시점을 가리키는 표현이면
  그 참조 대상 문장의 sent_id (역시 입력 형식 그대로). 그렇지 않으면 null.

# 입력 문서

{document}

# 출력 (JSON)

다른 설명 없이 JSON만 출력하세요.
"""


# Structured Outputs 강제 — JSON 파싱 실패 없음
# [v6.2] sent_id를 enum으로 동적 강제 → LLM이 prefix 누락 불가
_TEMPORAL_OUTPUT_SCHEMA_TEMPLATE: dict[str, Any] = {
    "type": "object",
    "properties": {
        "anchor_year": {
            "type": ["integer", "null"],
            "description": "문서의 기준 연도. 명시되지 않았으면 null."
        },
        "anchor_evidence": {
            "type": "string",
            "description": "anchor 추출 근거가 된 문장 또는 짧은 메모."
        },
        "temporal_expressions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sent_id": {"type": "string"},  # ← 빌드 시 enum으로 교체
                    "expression": {"type": "string"},
                    "resolved": {"type": ["string", "null"]},
                    "resolution_basis": {"type": "string"},
                    "refers_to_sent_id": {"type": ["string", "null"]},  # ← 동일
                },
                "required": ["sent_id", "expression", "resolved", "resolution_basis"],
            },
        },
    },
    "required": ["anchor_year", "anchor_evidence", "temporal_expressions"],
}


def _build_output_schema(valid_sent_ids: list[str]) -> dict[str, Any]:
    """
    [v6.2] 유효한 sent_id만 받도록 enum 강제하는 schema 동적 생성.

    Structured Outputs(HCX-007)이 enum 위반 응답을 거부하므로,
    LLM이 "b0001_s0003"을 "s0003"으로 줄여 답하는 것 자체가 불가능.
    """
    schema = copy.deepcopy(_TEMPORAL_OUTPUT_SCHEMA_TEMPLATE)
    items_props = schema["properties"]["temporal_expressions"]["items"]["properties"]

    # sent_id: 반드시 valid 값 중 하나
    items_props["sent_id"] = {
        "type": "string",
        "enum": valid_sent_ids,
        "description": "입력에 [...]로 표시된 sent_id 중 정확히 하나를 그대로 복사하세요.",
    }
    # refers_to_sent_id: valid 값 또는 null
    items_props["refers_to_sent_id"] = {
        "type": ["string", "null"],
        "enum": valid_sent_ids + [None],
        "description": "참조 대상 sent_id 또는 null",
    }
    return schema


# ── 메인 빌더 ───────────────────────────────────────────────────────────────

async def build_document_temporal_graph(
    sir_doc: SIRDocument,
    config: dict | None = None,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """
    문서 레벨 LLM agent 1회 호출 → 시간 그래프 빌드.

    [v6.2] sent_id를 Structured Outputs schema에서 enum으로 강제.
           LLM이 잘못된 sent_id를 답할 수 없음.

    Returns:
        ([DocumentNode, TemporalExprNodes, ResolvedTimeNodes],
         [HAS_TEMPORAL, RELATIVE_TO, RESOLVES_TO, REFERS_TO 엣지])
    """
    config = config or {}
    llm = LLMClient(config=config.get("llm", {}))

    doc_text = _format_document_with_ids(sir_doc)
    if not doc_text.strip():
        logger.warning("temporal graph: 빈 문서 — skip")
        return [], []

    # [v6.2] valid sent_id 목록을 추출하여 schema에 enum으로 박음
    valid_sent_ids = [
        sent.sent_id
        for block in sir_doc.blocks
        for sent in block.sentences
        if sent.sent_id
    ]
    if not valid_sent_ids:
        logger.warning("temporal graph: 유효한 sent_id 없음 — skip")
        return [], []

    output_schema = _build_output_schema(valid_sent_ids)

    try:
        result = await llm.generate_structured(
            prompt=_TEMPORAL_AGENT_PROMPT.format(document=doc_text),
            schema=output_schema,
            system_prompt=(
                "문서 시간 분석 전문가. 모든 시간 표현을 빠짐없이 수집하고, "
                "anchor와 문장 간 참조를 통해 가능한 한 절대 시점으로 풀어내세요. "
                "JSON으로만 답하세요."
            ),
        )
    except Exception as e:
        logger.warning(f"temporal agent 호출 실패: {e}")
        return [], []

    return _materialize_graph(sir_doc, result)


# ── 그래프 materialize ──────────────────────────────────────────────────────

def _materialize_graph(
    sir_doc: SIRDocument,
    agent_result: dict[str, Any],
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """LLM 출력 JSON → GraphNode/GraphEdge 변환."""
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    # 1) DocumentNode
    doc_node_id = f"node:doc:{sir_doc.doc_id.hex[:8]}"
    anchor_year = agent_result.get("anchor_year")
    nodes.append(GraphNode(
        node_id=doc_node_id,
        node_type=GraphNodeType.DOCUMENT,
        label=f"Document(anchor_year={anchor_year})",
        properties={
            "anchor_year": anchor_year,
            "anchor_evidence": agent_result.get("anchor_evidence"),
            "source_uri": sir_doc.source_uri,
            "source_type": sir_doc.source_type.value if sir_doc.source_type else None,
        },
    ))

    # sent_id → graph_anchor_id 매핑 (sentence 노드 id 검증용)
    sent_id_to_anchor = _build_sent_anchor_map(sir_doc)

    # 2) TemporalExprNodes + 엣지들
    seen_resolved: set[str] = set()
    expressions = agent_result.get("temporal_expressions", []) or []

    skipped_unmatched = 0
    skipped_ambiguous = 0

    for i, te in enumerate(expressions):
        sent_id = te.get("sent_id")
        expression = te.get("expression", "")
        resolved = te.get("resolved")
        basis = te.get("resolution_basis", "") or ""
        refers_to = te.get("refers_to_sent_id")

        if not sent_id or not expression:
            continue

        # 정확 매칭 → 실패 시 suffix 매칭 (LLM이 prefix 누락하는 경우 방어)
        sent_anchor = _resolve_sent_id(sent_id, sent_id_to_anchor)
        if sent_anchor is None:
            skipped_unmatched += 1
            logger.warning(
                f"temporal: sent_id={sent_id!r} 매칭 실패 (정확/suffix 모두) — skip"
            )
            continue
        if sent_anchor == "AMBIGUOUS":
            skipped_ambiguous += 1
            logger.warning(
                f"temporal: sent_id={sent_id!r} 모호 (suffix 매칭 다수) — skip"
            )
            continue

        # TemporalExpr 노드
        te_node_id = f"node:temporal:{sent_id}:{i}"
        nodes.append(GraphNode(
            node_id=te_node_id,
            node_type=GraphNodeType.TEMPORAL_EXPR,
            label=expression,
            properties={
                "sent_id": sent_id,
                "expression": expression,
                "resolution_basis": basis,
                "resolved_value": resolved,
            },
        ))

        # Sentence ─HAS_TEMPORAL→ TemporalExpr
        edges.append(GraphEdge(
            from_node=sent_anchor,
            to_node=te_node_id,
            edge_type=GraphEdgeType.HAS_TEMPORAL,
        ))

        # TemporalExpr ─RELATIVE_TO→ Document (anchor에 의존하는 표현)
        # LLM이 resolution_basis에 anchor를 언급했는지로 판단
        # (룰 매핑이 아니라 LLM이 자체 판단한 결과를 그래프에 박음)
        if anchor_year is not None and "anchor" in basis.lower():
            edges.append(GraphEdge(
                from_node=te_node_id,
                to_node=doc_node_id,
                edge_type=GraphEdgeType.RELATIVE_TO,
            ))

        # TemporalExpr ─RESOLVES_TO→ ResolvedTime
        if resolved:
            rt_node_id = f"node:resolved:{resolved}"
            if rt_node_id not in seen_resolved:
                nodes.append(GraphNode(
                    node_id=rt_node_id,
                    node_type=GraphNodeType.RESOLVED_TIME,
                    label=resolved,
                    properties={"value": resolved},
                ))
                seen_resolved.add(rt_node_id)
            edges.append(GraphEdge(
                from_node=te_node_id,
                to_node=rt_node_id,
                edge_type=GraphEdgeType.RESOLVES_TO,
            ))

        # TemporalExpr ─REFERS_TO→ 다른 Sentence (coreference)
        # 예: "재작년 같은 기간" → 앞 문장 "작년 1∼11월"을 참조
        if refers_to:
            ref_anchor = _resolve_sent_id(refers_to, sent_id_to_anchor)
            if ref_anchor and ref_anchor != "AMBIGUOUS":
                edges.append(GraphEdge(
                    from_node=te_node_id,
                    to_node=ref_anchor,
                    edge_type=GraphEdgeType.REFERS_TO,
                ))

    logger.info(
        f"document temporal graph: anchor_year={anchor_year}, "
        f"temporal_expressions={len(expressions)} "
        f"(skipped_unmatched={skipped_unmatched}, skipped_ambiguous={skipped_ambiguous}), "
        f"nodes={len(nodes)}, edges={len(edges)}"
    )
    return nodes, edges


# ── 헬퍼 ────────────────────────────────────────────────────────────────────

def _resolve_sent_id(
    sent_id: str,
    sent_id_to_anchor: dict[str, str],
) -> str | None:
    """
    LLM이 답한 sent_id를 sir_doc의 실제 sent_id에 매칭.

    1) 정확 매칭 우선
    2) 실패 시 suffix 매칭 (LLM이 "b0001_s0003"을 "s0003"으로 줄여 답한 경우)
    3) suffix 매칭이 여러 개면 모호 → "AMBIGUOUS" 반환 (skip 시그널)
    4) 매칭 안 되면 None

    Returns:
        graph_anchor_id (str), "AMBIGUOUS", 또는 None
    """
    # 1) 정확 매칭
    direct = sent_id_to_anchor.get(sent_id)
    if direct:
        return direct

    # 2) suffix 매칭 — block prefix 없이 답한 경우 ("s0003" ← "b0001_s0003")
    candidates = [
        v for k, v in sent_id_to_anchor.items()
        if k.endswith(f"_{sent_id}") or k == sent_id
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return "AMBIGUOUS"  # 여러 block에 같은 suffix → 어느 거 가리키는지 모름
    return None

def _format_document_with_ids(sir_doc: SIRDocument) -> str:
    """LLM 입력용으로 문서를 [sent_id] 접두사와 함께 포맷."""
    lines = []
    for block in sir_doc.blocks:
        for sent in block.sentences:
            text = sent.text.strip()
            if text:
                lines.append(f"[{sent.sent_id}] {text}")
    return "\n".join(lines)


def _build_sent_anchor_map(sir_doc: SIRDocument) -> dict[str, str]:
    """sent_id → graph_anchor_id 매핑."""
    mapping = {}
    for block in sir_doc.blocks:
        for sent in block.sentences:
            if sent.graph_anchor_id:
                mapping[sent.sent_id] = sent.graph_anchor_id
    return mapping