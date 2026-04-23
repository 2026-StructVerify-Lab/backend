"""
preprocessing/sir_builder.py — SIR Tree 빌더 (v2: Graph Anchor 포함)

기존 SIR JSON에서 SIR Tree로 확장. 각 블록/문장에 entity_refs, event_refs,
graph_anchor_ids를 부여하여 Graph Knowledge Layer와 연결 가능하게 한다.

[참고] Docling (IBM Research, 2024) — https://github.com/DS4SD/docling
  DoclingDocument의 block 단위 구조화 + 메타데이터 부여 방식 참고
"""
from __future__ import annotations
import re
from structverify.core.schemas import (
    BlockType, SIRBlock, SIRDocument, SourceOffset, SourceType)
from structverify.preprocessing.segmenter import split_sentences
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


def build_sir(raw_text: str, source_type: SourceType,
              source_uri: str | None = None) -> SIRDocument:
    """
    원시 텍스트 → SIR Tree 변환
    v2: 각 블록에 graph_anchor_ids, entity_refs 부여
    """
    blocks: list[SIRBlock] = []
    paragraphs = re.split(r"\n{2,}", raw_text)
    offset = 0

    for idx, para in enumerate(paragraphs):
        if not para.strip():
            offset += len(para) + 1
            continue

        block_type, level = _detect_block_type(para)
        sentences = split_sentences(para)
        start = raw_text.find(para, offset)
        anchor_id = f"node:b{idx:04d}"

        block = SIRBlock(
            block_id=f"b{idx:04d}", type=block_type, level=level,
            content=para, sentences=sentences,
            graph_anchor_ids=[anchor_id],
            entity_refs=_extract_entity_refs(para),   # v2
            event_refs=_extract_event_refs(para),      # v2
            source_offset=SourceOffset(char_start=max(start, 0),
                                        char_end=max(start, 0) + len(para)),
        )
        blocks.append(block)
        offset = block.source_offset.char_end

    doc = SIRDocument(source_type=source_type, source_uri=source_uri, blocks=blocks)
    logger.info(f"SIR Tree: {len(blocks)} blocks, "
                f"{sum(len(b.sentences) for b in blocks)} sentences")
    return doc


def _detect_block_type(text: str) -> tuple[BlockType, int | None]:
    m = re.match(r"^(#{1,6})\s+", text.strip())
    if m:
        return BlockType.HEADING, len(m.group(1))
    # TODO: 테이블/리스트 감지 로직 추가
    return BlockType.PARAGRAPH, None


def _extract_entity_refs(text: str) -> list[str]:
    """
    텍스트에서 엔티티 참조를 추출한다.
    TODO: spaCy NER 또는 LLM 기반 엔티티 추출 구현
    TODO: 도메인 팩의 Entity Dictionary와 매칭
    """
    return []


def _extract_event_refs(text: str) -> list[str]:
    """
    텍스트에서 이벤트/시점 참조를 추출한다.
    TODO: 시점 표현 정규식 (2024년, 전년 대비, 3분기 등) 파싱
    TODO: LLM 기반 temporal expression 추출
    """
    return []
