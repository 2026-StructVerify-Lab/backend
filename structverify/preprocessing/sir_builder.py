"""
preprocessing/sir_builder.py — SIR Tree 빌더 (v2: Graph Anchor 포함)

기존 SIR JSON에서 SIR Tree로 확장. 각 블록/문장에 entity_refs, event_refs,
graph_anchor_ids를 부여하여 Graph Knowledge Layer와 연결 가능하게 한다.

[참고] Docling (IBM Research, 2024) — https://github.com/DS4SD/docling
  DoclingDocument의 block 단위 구조화 + 메타데이터 부여 방식 참고


[김예슬]
- sentence offset을 block 상대 위치가 아니라 문서 전체 absolute offset으로 보정
- split_sentences()의 상대 offset을 block source_offset에 더해 절대 위치로 변환
"""
from __future__ import annotations

import re

from structverify.core.schemas import (
    BlockType,
    SIRBlock,
    SIRDocument,
    SourceOffset,
    SourceType,
)
from structverify.preprocessing.segmenter import split_sentences
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


def build_sir(
    raw_text: str,
    source_type: SourceType,
    source_uri: str | None = None,
) -> SIRDocument:
    """
    원시 텍스트 → SIR Tree 변환

    핵심 역할
    - 문단 분할
    - block metadata 생성
    - sentence offset 절대 위치 보정
    """
    blocks: list[SIRBlock] = []
    paragraphs = re.split(r"\n{2,}", raw_text)
    search_offset = 0

    for idx, para in enumerate(paragraphs):
        if not para.strip():
            search_offset += len(para) + 2
            continue

        block_type, level = _detect_block_type(para)
        block_start = raw_text.find(para, search_offset)
        if block_start < 0:
            block_start = search_offset
        block_end = block_start + len(para)

        sentences = split_sentences(para)

        # ★ 중요: 문장 offset을 문서 전체 기준 absolute offset으로 변환
        for sent in sentences:
            sent.char_offset_start = block_start + sent.char_offset_start
            sent.char_offset_end = block_start + sent.char_offset_end

        anchor_id = f"node:b{idx:04d}"
        block = SIRBlock(
            block_id=f"b{idx:04d}",
            type=block_type,
            level=level,
            content=para,
            sentences=sentences,
            graph_anchor_ids=[anchor_id],
            entity_refs=_extract_entity_refs(para),
            event_refs=_extract_event_refs(para),
            source_offset=SourceOffset(
                char_start=max(block_start, 0),
                char_end=max(block_end, 0),
            ),
        )
        blocks.append(block)
        search_offset = block_end

    return SIRDocument(
        source_type=source_type,
        source_uri=source_uri,
        blocks=blocks,
    )


def _detect_block_type(text: str) -> tuple[BlockType, int | None]:
    """
    아주 단순한 블록 타입 판별기.
    필요 시 heading/list/table 휴리스틱은 더 보강 가능.
    """
    stripped = text.strip()

    if re.match(r"^#+\s+", stripped):
        return BlockType.HEADING, stripped.count("#", 0, stripped.find(" "))
    if stripped.startswith("- ") or stripped.startswith("* "):
        return BlockType.LIST, None
    if "\t" in stripped or "|" in stripped:
        return BlockType.TABLE, None
    return BlockType.PARAGRAPH, None


def _extract_entity_refs(text: str) -> list[str]:
    """
    placeholder 수준의 entity reference 추출.
    나중에 NER 또는 dictionary 매핑으로 교체 가능.
    """
    refs: list[str] = []
    for token in re.findall(r"[가-힣A-Za-z0-9]{2,}", text):
        if token.endswith(("청", "부", "국", "원", "시", "군", "구")):
            refs.append(f"entity:{token}")
    return list(dict.fromkeys(refs))


def _extract_event_refs(text: str) -> list[str]:
    """
    시점/이벤트 비슷한 표현만 약하게 추출.
    """
    refs: list[str] = []
    for m in re.findall(r"\d{4}년|\d+월|\d+분기|전년|지난해|올해", text):
        refs.append(f"event:{m}")
    return list(dict.fromkeys(refs))