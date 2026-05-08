"""
preprocessing/docx/pipeline.py — DOCX → Markdown 오케스트레이터

단계:
  1) python-docx 파싱 (`reader.parse_docx`)
  2) Docling 보강 시도 → title/date 덮어쓰기 (선택적)
  3) title / date 폴백 처리
  4) Markdown 직렬화 (`markdown.to_markdown`)

Docling 미설치 또는 변환 실패 시 python-docx 결과만으로 계속 진행.
"""
from __future__ import annotations

from structverify.preprocessing.docx import reader, markdown as md_mod
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


def extract_docx_to_markdown(filepath: str) -> str:
    """DOCX 파일 경로 → Markdown 문자열. 공개 API."""
    try:
        ex = reader.parse_docx(filepath)
    except Exception as e:
        logger.error(f"DOCX 열기 실패: {filepath} — {e}")
        return ""

    # Docling 보강: title이 비어있을 때만 시도
    if not ex.title:
        dl = reader.docling_extract(filepath)
        if dl["ok"] and dl["json"]:
            dl_title = (dl["json"].get("title") or "").strip()
            if dl_title:
                ex.title = dl_title

    if not ex.paragraphs and not ex.tables:
        logger.warning(f"DOCX 내용 없음: {filepath}")
        return ""

    return md_mod.to_markdown(ex)