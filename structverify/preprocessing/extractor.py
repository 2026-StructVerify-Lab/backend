"""
preprocessing/extractor.py — 소스 유형별 텍스트 추출기

[참고] Trafilatura (Barbaresi, ACL 2021) — https://github.com/adbar/trafilatura
  논문: Barbaresi, A. (2021). Trafilatura: A Web Scraping Library and
  Command-Line Tool for Text Discovery and Extraction. ACL 2021.
  URL 입력 시 광고/네비게이션 제거 후 본문만 추출

[참고] Docling (IBM Research, 2024) — https://github.com/DS4SD/docling
  PDF/DOCX를 통합 DoclingDocument JSON으로 변환. TableFormer로 테이블 구조 인식.

[참고] PyMuPDF — https://github.com/pymupdf/PyMuPDF
  PDF 텍스트/테이블 페이지 단위 추출

[참고] python-docx — https://github.com/python-openxml/python-docx
  DOCX XML 구조 파싱
"""
from __future__ import annotations
from structverify.core.schemas import SourceType
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


async def extract_text(source: str, source_type: SourceType) -> str:
    """소스 유형에 따라 적절한 추출기를 호출하여 텍스트 반환"""
    if source_type == SourceType.URL:
        return await _extract_from_url(source)
    elif source_type == SourceType.PDF:
        return _extract_from_pdf(source)
    elif source_type == SourceType.DOCX:
        return _extract_from_docx(source)
    elif source_type == SourceType.TEXT:
        return source
    raise ValueError(f"미지원 소스: {source_type}")


async def _extract_from_url(url: str) -> str:
    """
    URL → 본문 텍스트 추출 (Trafilatura)
    TODO: trafilatura.fetch_url() + trafilatura.extract(include_tables=True) 구현
    """
    logger.warning(f"URL 추출 stub: {url}")
    return f"[STUB] {url}"


def _extract_from_pdf(filepath: str) -> str:
    """
    PDF → 페이지별 텍스트 추출 (PyMuPDF / Docling)
    TODO: fitz.open(filepath) → page.get_text("text") 구현
    TODO: 스캔 PDF OCR 폴백 (Tesseract / EasyOCR)
    TODO: Docling 연동으로 테이블 구조 인식 강화
    """
    logger.warning(f"PDF 추출 stub: {filepath}")
    return f"[STUB] {filepath}"


def _extract_from_docx(filepath: str) -> str:
    """
    DOCX → 문단/테이블 추출 (python-docx)
    TODO: Document(filepath).paragraphs + .tables 파싱 구현
    """
    logger.warning(f"DOCX 추출 stub: {filepath}")
    return f"[STUB] {filepath}"
