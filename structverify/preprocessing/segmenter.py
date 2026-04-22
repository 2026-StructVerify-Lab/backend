"""
preprocessing/segmenter.py — 한국어 문장 분리 + surface signal 추출

중요 변경
- regex는 최종 검증 후보 판단기가 아니다.
- 정규식은 has_numeric_surface 같은 약한 signal만 생성한다.
- 실제 candidate 판단은 detection/candidate_scorer.py에서 수행한다.
"""
from __future__ import annotations

import re

from structverify.core.schemas import Sentence
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

# 숫자/단위/비율/수량 표현용 약한 표면 신호
NUMERIC_SURFACE_PATTERN = re.compile(
    r"""
    (
        \d[\d,.\-]*\s*(%|퍼센트|원|만원|억원|조원|ha|km|m²|건|명|가구|개|톤|배럴|배)?
        |
        \d{4}년|\d+월|\d+분기
        |
        전년\s*대비|지난해보다|증가|감소|상승|하락|올랐다|내렸다
    )
    """,
    re.UNICODE | re.VERBOSE,
)


def split_sentences(text: str) -> list[Sentence]:
    """
    텍스트 → Sentence 리스트

    역할
    - 문장 분리
    - 문장별 상대 offset 계산
    - surface signal(has_numeric_surface) 부여
    - candidate_score / candidate_label은 기본값만 세팅

    주의
    - 여기서는 "검증 후보"를 결정하지 않는다.
    - 실제 후보 판단은 candidate scorer가 맡는다.
    """
    raw = _split_korean(text)
    sentences: list[Sentence] = []
    offset = 0
    seq = 0

    for s in raw:
        s = s.strip()
        if not s:
            continue

        start = text.find(s, offset)
        if start < 0:
            start = offset
        end = start + len(s)

        anchor_id = f"node:s{seq:04d}"
        surface_hit = bool(NUMERIC_SURFACE_PATTERN.search(s))

        sentences.append(
            Sentence(
                sent_id=f"s{seq:04d}",
                text=s,
                char_offset_start=max(start, 0),
                char_offset_end=max(end, 0),
                has_numeric_surface=surface_hit,
                candidate_score=0.0,
                candidate_label=False,
                candidate_source="surface_rule",
                candidate_signals={
                    "surface_numeric_hit": surface_hit,
                },
                graph_anchor_id=anchor_id,
            )
        )

        offset = end
        seq += 1

    return sentences


def _split_korean(text: str) -> list[str]:
    """
    kss 우선 사용, 없으면 정규표현식 폴백.

    운영 환경에서 kss를 쓰는 것을 권장한다.
    """
    try:
        import kss
        return kss.split_sentences(text)
    except ImportError:
        logger.warning("kss 미설치 — 정규표현식 폴백 사용")
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]