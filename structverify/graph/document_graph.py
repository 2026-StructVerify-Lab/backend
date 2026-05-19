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

import re
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
- ⚠️ **최우선 규칙**: 본문에 기사 작성일/발행일이 있으면 (예: "작성일자 2025-01-01",
  "2025.01.01", "입력 2025-01-01 11:34", "발행: 2025년 1월 1일") **그 날짜의 연도를
  anchor로 무조건 선택**한다. 본문 내용("작년 기온이 가장 높았다" 등)이 다른 연도를
  암시하더라도 작성일 연도가 절대 우선이다. "작년/지난해"는 작성일 연도 - 1 이다.
- 작성일이 없을 때만: 본문에 "OOOO년은", "지난 OOOO년" 등으로 명시된 연도를 anchor로
- 명시된 anchor가 여러 개라면 본문 서술 시점의 기준이 되는 것을 선택
- 본문에 명시되지 않은 경우 anchor_year=null
- 무엇을 근거로 anchor를 정했는지 anchor_evidence에 짧게 기록
  (작성일을 썼다면 "작성일자 2025-01-01 → anchor_year=2025"처럼 기록)

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
_TEMPORAL_OUTPUT_SCHEMA: dict[str, Any] = {
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
                    "sent_id": {"type": "string"},
                    "expression": {"type": "string"},
                    "resolved": {"type": ["string", "null"]},
                    "resolution_basis": {"type": "string"},
                    "refers_to_sent_id": {"type": ["string", "null"]},
                },
                "required": ["sent_id", "expression", "resolved", "resolution_basis"],
            },
        },
    },
    "required": ["anchor_year", "anchor_evidence", "temporal_expressions"],
}


# ── 메인 빌더 ───────────────────────────────────────────────────────────────

async def build_document_temporal_graph(
    sir_doc: SIRDocument,
    config: dict | None = None,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """
    문서 레벨 LLM agent 1회 호출 → 시간 그래프 빌드.

    Args:
        sir_doc: SIR Document (blocks/sentences 포함)
        config: LLM 설정

    Returns:
        ([DocumentNode, TemporalExprNodes, ResolvedTimeNodes],
         [HAS_TEMPORAL, RELATIVE_TO, RESOLVES_TO, REFERS_TO 엣지])

    Note:
        - SentenceNode/BlockNode는 graph_builder.py가 만드는 기존 노드를 재사용
          (sent.graph_anchor_id 그대로 참조)
        - LLM 1회 호출만으로 전체 문서의 시간 정보 추출
        - 실패 시 빈 리스트 반환 — 다운스트림은 anchor 정보 없이 작동
    """
    config = config or {}
    llm = LLMClient(config=config.get("llm", {}))

    doc_text = _format_document_with_ids(sir_doc)
    if not doc_text.strip():
        logger.warning("temporal graph: 빈 문서 — skip")
        return [], []

    try:
        result = await llm.generate_structured(
            prompt=_TEMPORAL_AGENT_PROMPT.format(document=doc_text),
            schema=_TEMPORAL_OUTPUT_SCHEMA,
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
    anchor_evidence = agent_result.get("anchor_evidence")

    # ── [v6.19] 작성일 기반 anchor 덮어쓰기 ──────────────────────────────
    # LLM이 본문 내용에 끌려 anchor를 잘못 잡을 수 있어, 본문에 명시된
    # 작성일/발행일 연도가 있으면 그것을 anchor로 강제한다.
    article_year = _extract_article_year(sir_doc)
    anchor_corrected = False
    if article_year is not None and article_year != anchor_year:
        logger.info(
            f"temporal graph: anchor_year를 작성일 기준으로 보정 "
            f"{anchor_year} → {article_year} (LLM 추론값 무시)"
        )
        anchor_year = article_year
        anchor_corrected = True
        anchor_evidence = (
            f"본문 작성일 기준 anchor_year={article_year} "
            f"(LLM 추론값 대신 보정)"
        )

    nodes.append(GraphNode(
        node_id=doc_node_id,
        node_type=GraphNodeType.DOCUMENT,
        label=f"Document(anchor_year={anchor_year})",
        properties={
            "anchor_year": anchor_year,
            "anchor_evidence": anchor_evidence,
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

        # [v6.19] anchor가 작성일 기준으로 보정됐으면 상대 표현 resolved도 재계산
        if anchor_corrected and anchor_year is not None:
            new_resolved = _recompute_resolved(
                expression, basis, resolved, anchor_year
            )
            if new_resolved != resolved:
                logger.info(
                    f"temporal: '{expression}' resolved 재계산 "
                    f"{resolved!r} → {new_resolved!r} (anchor={anchor_year})"
                )
                resolved = new_resolved

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

# ── [v6.19] 작성일 기반 anchor 보조 가드 ──────────────────────────────────
# LLM이 본문 내용("작년 가장 더웠다")에 끌려 anchor를 잘못 잡는 경우가 있어
# (예: 작성일 2025인데 anchor=2024), 본문에 명시된 작성일/발행일 연도를
# deterministic하게 추출해 anchor를 덮어쓴다.

# "작성일자 2025-01-01", "입력 2025.01.01 11:34", "2025-01-01 11:34" 등
_DATE_PATTERNS = [
    re.compile(r"(20\d{2})\s?[.\-/년]\s?\d{1,2}\s?[.\-/월]\s?\d{1,2}"),
]
# 작성일 맥락 키워드 — 이 단어 근처의 날짜만 작성일로 인정
_DATE_CONTEXT = ("작성", "발행", "입력", "등록", "송고", "보도")


def _extract_article_year(sir_doc: SIRDocument) -> int | None:
    """본문에서 기사 작성일/발행일의 연도를 추출.

    작성일 맥락 키워드(_DATE_CONTEXT)가 같은 문장에 있는 날짜를 우선.
    맥락 키워드가 없으면, 본문 맨 앞쪽(상위 3문장)의 날짜를 후보로 본다
    (기사 헤더에 날짜만 단독으로 오는 경우 대응).
    없으면 None — LLM이 뽑은 anchor를 그대로 둔다.
    """
    context_year: int | None = None
    sent_idx = 0
    for block in sir_doc.blocks:
        for sent in block.sentences:
            text = (sent.text or "").strip()
            sent_idx += 1
            if not text:
                continue
            for pat in _DATE_PATTERNS:
                m = pat.search(text)
                if not m:
                    continue
                year = int(m.group(1))
                if not (2000 <= year <= 2099):
                    continue
                has_ctx = any(kw in text for kw in _DATE_CONTEXT)
                if has_ctx:
                    return year  # 작성일 맥락 — 즉시 확정
                if context_year is None and sent_idx <= 3:
                    context_year = year  # 헤더 부근 날짜 — 후보로 보관
    return context_year


# ── [v6.19] anchor 보정 시 상대 시간표현 resolved 재계산 ──────────────────
# anchor를 작성일 기준으로 덮어쓰면, LLM이 옛 anchor로 풀어둔 resolved 값
# ("작년"→2023)이 어긋난다. resolution_basis의 offset(anchor-1 등)을 읽어
# 새 anchor 기준으로 다시 계산한다.

# "작년/지난해/전년" = -1, "재작년" = -2, "올해/금년" = 0, "내년" = +1
_RELATIVE_OFFSETS = {
    "재작년": -2, "지지난해": -2,
    "작년": -1, "지난해": -1, "전년": -1, "지난 해": -1,
    "올해": 0, "금년": 0, "당해": 0, "올 해": 0,
    "내년": 1, "명년": 1,
}


def _offset_from_basis(basis: str) -> int | None:
    """resolution_basis 문자열에서 anchor 대비 offset을 추출.

    "anchor_year - 1 = 2023" → -1, "anchor_year-2(2022)" → -2,
    "anchor_year" 단독 → 0. 못 찾으면 None.
    """
    if not basis or "anchor" not in basis.lower():
        return None
    m = re.search(r"anchor[_ ]?year\s*([+\-])\s*(\d+)", basis, re.IGNORECASE)
    if m:
        sign = -1 if m.group(1) == "-" else 1
        return sign * int(m.group(2))
    # offset 표기 없이 anchor_year만 언급 → 같은 해
    return 0


def _recompute_resolved(
    expression: str, basis: str, resolved: Any, new_anchor: int,
) -> Any:
    """상대 시간표현의 resolved를 new_anchor 기준으로 재계산.

    - expression이 상대 표현(_RELATIVE_OFFSETS)이면 그 offset 우선 사용
    - 아니면 resolution_basis의 anchor offset 사용
    - 둘 다 없으면 원본 resolved 유지 (절대 표현 "2024년" 등은 건드리지 않음)
    """
    expr = (expression or "").strip()
    offset: int | None = None
    for kw, off in _RELATIVE_OFFSETS.items():
        if kw in expr:
            offset = off
            break
    if offset is None:
        offset = _offset_from_basis(basis)
    if offset is None:
        return resolved  # 절대 표현 — 그대로

    new_year = new_anchor + offset
    # resolved가 'YYYY-MM' / 'YYYY-MM..YYYY-MM' 형태면 연도 부분만 치환
    if isinstance(resolved, str) and resolved:
        # 연도(YYYY)를 new_year로 교체, 월/범위는 유지
        return re.sub(r"20\d{2}", str(new_year), resolved)
    return str(new_year)