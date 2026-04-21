"""
preprocessing/segmenter.py — 한국어 문장 분리 + 수치 탐지

[참고] kss (Korean Sentence Splitter) — https://github.com/hyunwoongko/kss
  한국어 특화 문장 분리. 약어/수치 포함 문장에서도 높은 정확도.
"""
from __future__ import annotations
import re
from structverify.core.schemas import Sentence
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

NUMERIC_PATTERN = re.compile(
    r"\d[\d,.]*\s*(%|퍼센트|원|만|억|조|ha|km|m²|건|명|가구|개|톤|배)", re.UNICODE)


def split_sentences(text: str) -> list[Sentence]:
    """텍스트 → 문장 리스트 (수치 포함 여부 태깅 + 그래프 앵커 ID 부여)"""
    raw = _split_korean(text)
    sentences, offset = [], 0
    for idx, s in enumerate(raw):
        s = s.strip()
        if not s:
            continue
        start = text.find(s, offset)
        end = start + len(s) if start >= 0 else offset + len(s)
        anchor_id = f"node:s{idx:04d}"
        sentences.append(Sentence(
            sent_id=f"s{idx:04d}", text=s,
            char_offset_start=max(start, 0), char_offset_end=end,
            has_numeric=bool(NUMERIC_PATTERN.search(s)),
            graph_anchor_id=anchor_id,
        ))
        offset = end
    return sentences


def _split_korean(text: str) -> list[str]:
    """kss 우선 사용, 없으면 정규표현식 폴백"""
    try:
        import kss
        return kss.split_sentences(text)
    except ImportError:
        logger.warning("kss 미설치 — 정규표현식 폴백")
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
