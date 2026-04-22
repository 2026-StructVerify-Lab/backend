"""
preprocessing/sir_builder.py — SIR Tree 빌더 (v2: Graph Anchor 포함)

기존 SIR JSON에서 SIR Tree로 확장. 각 블록/문장에 entity_refs, event_refs,
graph_anchor_ids를 부여하여 Graph Knowledge Layer와 연결 가능하게 한다.

[이수민]
- 문단 분할 + SIRBlock 생성 + 문장 offset 절대 위치 보정 담당
- entity_refs 추출은 현재 regex placeholder → 추후 NER 모델로 교체 예정

[김예슬]
- sentence offset을 block 상대 위치가 아니라 문서 전체 absolute offset으로 보정
- split_sentences()의 상대 offset을 block source_offset에 더해 절대 위치로 변환

[참고] Docling (IBM Research, 2024) — https://github.com/DS4SD/docling
  DoclingDocument의 block 단위 구조화 + 메타데이터 부여 방식 참고
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

    TODO [이수민]: 문단 분할 로직 개선
      - 현재: re.split(r"\n{2,}") 단순 분할
      - 개선: heading 감지, 목록/표 블록 분리 정확도 향상
      - PDF 입력 시 페이지 경계 처리 (source_offset.page 활용)

    TODO [이수민]: entity_refs 추출 개선
      - 현재: 기관명 suffix(청/부/국 등) 매칭 → regex placeholder
      - 개선: NER 모델 또는 dictionary lookup으로 교체
      - 주요 대상: 기관명, 지역명, 통계 관련 고유명사
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
    블록 타입 판별기.

    TODO [이수민]: heading/list/table 감지 정확도 향상
      - Markdown 형식 외에 뉴스 기사 형식도 고려
      - 소제목 패턴 (예: "■ 농업 현황", "○ 주요 결과") 감지
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
    Entity reference 추출.

    TODO [이수민]: NER 모델 또는 dictionary lookup으로 교체
      현재 구현: 기관명 suffix 패턴 매칭 (regex placeholder)
      교체 계획:
        - spaCy 또는 klue/roberta 기반 NER 모델 사용
        - 또는 통계청/농림부 등 주요 기관명 사전 구축
      주의: NER 결과를 "entity:{이름}" 형태로 그대로 사용
    """
    refs: list[str] = []
    for token in re.findall(r"[가-힣A-Za-z0-9]{2,}", text):
        if token.endswith(("청", "부", "국", "원", "시", "군", "구")):
            refs.append(f"entity:{token}")
    return list(dict.fromkeys(refs))


def _extract_event_refs(text: str) -> list[str]:
    """
    시점/이벤트 표현 추출.

    TODO [이수민]: 시점 표현 다양성 대응
      현재: 연/월/분기/전년/지난해/올해 패턴만 감지
      개선: "1분기", "상반기", "하반기", "연간" 등 추가
    """
    refs: list[str] = []
    for m in re.findall(r"\d{4}년|\d+월|\d+분기|전년|지난해|올해", text):
        refs.append(f"event:{m}")
    return list(dict.fromkeys(refs))
