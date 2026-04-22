"""
preprocessing/segmenter.py — 한국어 문장 분리 + surface signal 추출

[이수민]
- kss 기반 한국어 문장 분리 구현 담당
- surface signal(has_numeric_surface)은 fallback 보조 신호만 생성
- 실제 candidate 판단은 detection/candidate_scorer.py의 LLM이 담당

[중요 설계 원칙]
- Regex는 최종 검증 후보 판단기가 아니다.
- 정규식은 has_numeric_surface 같은 약한 signal만 생성한다.
- 실제 candidate 판단은 detection/candidate_scorer.py에서 LLM으로 수행한다.
- Regex 기반 rule로 검증 후보를 결정하지 않도록 주의.
"""
from __future__ import annotations

import re

from structverify.core.schemas import Sentence
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

# ── surface signal용 약한 정규식 (fallback 보조 신호만) ──────────────────
# 이 패턴은 candidate_scorer의 LLM 판단이 불가능할 때만 사용하는 fallback입니다.
# Rule로 검증 후보를 결정하지 마세요 — LLM이 최종 판단합니다.
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
    - 문장 분리 (kss 우선, 정규식 폴백)
    - 문장별 상대 offset 계산
    - surface signal(has_numeric_surface) 부여 — fallback용 약한 신호
    - candidate_score / candidate_label은 기본값만 세팅

    주의
    - 여기서는 "검증 후보"를 결정하지 않는다.
    - 실제 후보 판단은 candidate scorer(LLM)가 맡는다.

    TODO [이수민]: kss 설치 확인 및 split_sentences 동작 검증
      pip install kss
      - kss.split_sentences() 호출이 제대로 되는지 테스트
      - 긴 문단, 인용문, 줄임표 처리 엣지케이스 확인
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
        # surface_hit은 약한 보조 신호 — LLM 판단의 fallback용
        surface_hit = bool(NUMERIC_SURFACE_PATTERN.search(s))

        sentences.append(
            Sentence(
                sent_id=f"s{seq:04d}",
                text=s,
                char_offset_start=max(start, 0),
                char_offset_end=max(end, 0),
                has_numeric_surface=surface_hit,
                # candidate_score/label은 candidate_scorer.py에서 채워짐
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

    TODO [이수민]: kss 설치 후 동작 확인
      - import kss
      - kss.split_sentences(text) 반환값이 list[str]인지 확인
      - kss 버전에 따라 API가 다를 수 있음 (kss 4.x vs 6.x)

    운영 환경에서 kss를 쓰는 것을 권장한다.
    """
    try:
        import kss
        return kss.split_sentences(text)
    except ImportError:
        logger.warning("kss 미설치 — 정규표현식 폴백 사용")
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
