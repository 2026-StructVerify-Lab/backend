"""
preprocessing/sir_builder.py — SIR Tree 빌더 (v1: 기본 정제 + GraphRAG 문맥 엣지)

[김예슬 - 2026-04-24]
- _clean_text_basic() 추가:
  · URL 제거, 연속 빈줄 정리, 연속 공백 정리 (최소 정제만)
  · LLM 정제는 추후 use_llm_clean: true 옵션으로 활성화 예정
- 절대 char_offset 보정:
  · 기존: segmenter 상대 offset 그대로 사용 (버그)
  · 수정: block_start + 상대 offset → 절대 offset
- 전역 sent_id 부여:
  · 기존: s0000, s0001 (블록 내 중복 가능)
  · 수정: b0000s0000, b0001s0000 (문서 전체 유일)
- _detect_block_type(): LIST, TABLE 감지 추가
- extract_context_edges(): GraphRAG용 문맥 엣지 추출 추가
  · NEXT_SENT: 문장 간 순서 (문맥 흐름 보존)
  · IN_BLOCK: 문장 → 소속 문단
  · IN_DOC: 문단 → 소속 문서
  → graph_builder.py에서 Neo4j에 저장하면
    2-hop 탐색으로 "같은 기사 내 연관 수치" 발견 가능

[참고] Docling (IBM Research, 2024) — https://github.com/DS4SD/docling
"""
from __future__ import annotations

import re
from typing import Any

from structverify.core.schemas import (
    BlockType, SIRBlock, SIRDocument, SourceOffset, SourceType,
)
from structverify.preprocessing.segmenter import split_sentences
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

# ── 기본 정제 패턴 ────────────────────────────────────────────────────────
_URL_RE      = re.compile(r'https?://\S+')
_BLANK_LINES = re.compile(r'\n{3,}')
_SPACES      = re.compile(r'[ \t]{2,}')

# ════════════════════════════════════════════════════════════════════════
# 메인 빌더
# ════════════════════════════════════════════════════════════════════════

def build_sir(
    raw_text: str,
    source_type: SourceType,
    source_uri: str | None = None,
) -> SIRDocument:
    """
    MD 텍스트 → SIRDocument 변환.

    처리 순서:
      1) _clean_text_basic(): URL 제거, 빈줄 정리 (최소 정제)
      2) \\n{2,} 기준 문단 분할
      3) 블록 타입 감지 (HEADING / PARAGRAPH / LIST / TABLE)
      4) kss 문장 분리 + 절대 offset 보정
      5) graph_anchor_id 부여 (블록/문장 모두)

    LLM 기반 심층 정제가 필요하면 추후 build_sir_async() 사용.
    """
    cleaned = _clean_text_basic(raw_text)

    blocks: list[SIRBlock] = []
    search_pos = 0
    block_seq = 0

    for para in re.split(r'\n{2,}', cleaned):
        para_stripped = para.strip()

        # 빈 문단 스킵
        if not para_stripped or len(para_stripped) < 2:
            search_pos += len(para) + 2
            continue

        block_type, level = _detect_block_type(para_stripped)

        # 절대 시작 위치 계산
        block_start = cleaned.find(para_stripped, search_pos)
        if block_start < 0:
            block_start = search_pos
        block_end = block_start + len(para_stripped)

        block_id     = f"b{block_seq:04d}"
        block_anchor = f"node:{block_id}"

        # 문장 분리 + 절대 offset 보정
        sentences = split_sentences(para_stripped)
        for sent in sentences:
            # 전역 유일 sent_id (블록ID + 문장ID 조합)
            global_sent_id         = f"{block_id}_{sent.sent_id}"
            sent.sent_id           = global_sent_id
            sent.graph_anchor_id   = f"node:{global_sent_id}"
            # 상대 offset → 절대 offset
            sent.char_offset_start = block_start + sent.char_offset_start
            sent.char_offset_end   = block_start + sent.char_offset_end

        block = SIRBlock(
            block_id=block_id,
            type=block_type,
            level=level,
            content=para_stripped,
            sentences=sentences,
            graph_anchor_ids=[block_anchor],
            entity_refs=_extract_entity_refs(para_stripped),
            event_refs=_extract_event_refs(para_stripped),
            source_offset=SourceOffset(
                char_start=block_start,
                char_end=block_end,
            ),
        )
        blocks.append(block)
        search_pos = block_end
        block_seq += 1

    doc = SIRDocument(
        source_type=source_type,
        source_uri=source_uri,
        blocks=blocks,
    )
    total_sents = sum(len(b.sentences) for b in blocks)
    logger.info(f"SIR Tree: {len(blocks)} blocks, {total_sents} sentences [{source_type.value}]")
    return doc


# ════════════════════════════════════════════════════════════════════════
# GraphRAG 문맥 엣지
# ════════════════════════════════════════════════════════════════════════

def extract_context_edges(doc: SIRDocument) -> list[dict[str, Any]]:
    """
    GraphRAG용 문맥 엣지 추출.

    graph_builder.py에서 이 함수를 호출하여 Neo4j에 저장하면
    2-hop 탐색으로 "같은 기사 내 연관 수치"를 발견할 수 있다.

    엣지 유형:
      NEXT_SENT : 문장 → 다음 문장 (문맥 흐름 보존)
      IN_BLOCK  : 문장 → 소속 문단
      IN_DOC    : 문단 → 소속 문서

    TODO [이수민]: graph_builder.py에서 이 함수 호출 후 Neo4j 저장
    """
    edges: list[dict[str, Any]] = []
    doc_node = f"node:doc:{doc.doc_id.hex[:8]}"

    for block in doc.blocks:
        block_anchor = block.graph_anchor_ids[0] if block.graph_anchor_ids else None
        if not block_anchor:
            continue

        # IN_DOC: 문단 → 문서
        edges.append({
            "from_node": block_anchor,
            "to_node":   doc_node,
            "edge_type": "IN_DOC",
        })

        sents = block.sentences
        for i, sent in enumerate(sents):
            if not sent.graph_anchor_id:
                continue

            # IN_BLOCK: 문장 → 문단
            edges.append({
                "from_node": sent.graph_anchor_id,
                "to_node":   block_anchor,
                "edge_type": "IN_BLOCK",
            })

            # NEXT_SENT: 문장 → 다음 문장
            if i + 1 < len(sents) and sents[i + 1].graph_anchor_id:
                edges.append({
                    "from_node": sent.graph_anchor_id,
                    "to_node":   sents[i + 1].graph_anchor_id,
                    "edge_type": "NEXT_SENT",
                })

    logger.debug(f"문맥 엣지 {len(edges)}개 추출")
    return edges


# ════════════════════════════════════════════════════════════════════════
# 내부 유틸
# ════════════════════════════════════════════════════════════════════════

def _clean_text_basic(raw: str) -> str:
    """
    최소한의 공통 정제.
    어떤 소스든 확실히 해당되는 것만 처리.

    - URL 제거
    - 연속 공백(탭) → 단일 공백
    - 3줄 이상 빈 줄 → 2줄

    노이즈 심화 정제(페이지번호, 이미지섹션 등)는
    추후 LLM 기반 clean_text_with_llm()으로 처리 예정.
    """
    text = _URL_RE.sub('', raw)
    text = _SPACES.sub(' ', text)
    text = _BLANK_LINES.sub('\n\n', text)
    return text.strip()


def _detect_block_type(text: str) -> tuple[BlockType, int | None]:
    """MD 블록 타입 감지."""
    stripped = text.strip()

    # 헤딩: # ~ ######
    m = re.match(r'^(#{1,6})\s+', stripped)
    if m:
        return BlockType.HEADING, len(m.group(1))

    # 테이블: | 로 시작
    if stripped.startswith('|') and '|' in stripped:
        return BlockType.TABLE, None

    # 목록: -, *, 숫자.
    if re.match(r'^[-*]\s+', stripped) or re.match(r'^\d+\.\s+', stripped):
        return BlockType.LIST, None

    return BlockType.PARAGRAPH, None


def _extract_entity_refs(text: str) -> list[str]:
    """
    기관명/단체명 추출.
    현재 구현 안함.
    TODO : 김예슬
    """
    return []


def _extract_event_refs(text: str) -> list[str]:
    """날짜/시점 표현 추출.
    현재 구현 안함.
    TODO : 김예슬
    """
    return []