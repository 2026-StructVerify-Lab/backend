"""
preprocessing/pdf/pipeline.py — PDF → Markdown 오케스트레이터

단계:
  1) PyMuPDF 페이지 병렬 추출 (`reader.extract_all_pages`)
  2) Docling JSON+HTML 추출 (`reader.docling_extract`)
  3) 소스 스코어링 → 선택 (`scoring.pick_source`)
  4) 선택 소스에서 title/date/body 추출 + 평문 정규식 폴백 (`fields`)
  5) 이미지 블록 crop OCR 후 본문 인라인 삽입 (`ocr` + `markdown`)
  6) 스캔 페이지 전체 OCR 후 body 교체/추가 (`ocr` + `markdown`)
  7) Markdown 직렬화 (`markdown.to_markdown`)

실패에 관대하게: Docling/OCR 어느 단계가 고장나도 가능한 결과를 반환.
"""
from __future__ import annotations

from structverify.preprocessing.pdf.models import Extracted
from structverify.preprocessing.pdf import reader, scoring, fields, ocr, markdown
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


def extract_pdf_to_markdown(filepath: str,
                            ocr_backend: str | None = None) -> str:
    """PDF 파일 경로 → Markdown 문자열. 공개 API."""
    pdf_path = str(filepath)

    # 1) 페이지 텍스트
    try:
        pages = reader.extract_all_pages(pdf_path)
    except Exception as e:
        logger.error(f"PDF 열기 실패: {pdf_path} — {e}")
        return ""
    plain = "\n\n".join(p.text for p in pages)

    # 2~3) 구조화 추출 + 소스 선택
    dl = reader.docling_extract(pdf_path)
    if dl["ok"]:
        src = scoring.pick_source(dl)
        ex = (fields.extract_from_json(dl["json"]) if src == "json"
              else fields.extract_from_html(dl["html"]))
    else:
        ex = Extracted(body=plain, source_used="pymupdf")

    # 4) 필드 폴백
    if not ex.title:
        ex.title = fields.fallback_from_plain(plain, "title")
    if not ex.date:
        ex.date = fields.fallback_from_plain(plain, "date")
    if len(ex.body) < 50:
        ex.body = plain

    # 5) 이미지 블록 crop OCR → 본문 인라인
    try:
        image_ocr = ocr.collect_image_ocr(pdf_path, pages, ocr_backend)
    except Exception as e:
        logger.warning(f"이미지 OCR 단계 스킵: {e}")
        image_ocr = {}
    if image_ocr:
        ex.body = markdown.inline_image_ocr_into_body(ex.body, image_ocr, pages)

    # 6) 스캔 페이지 전체 OCR → body 교체/추가
    try:
        scanned_ocr = ocr.collect_scanned_ocr(pdf_path, pages, ocr_backend)
    except Exception as e:
        logger.warning(f"스캔 OCR 단계 스킵: {e}")
        scanned_ocr = {}
    if scanned_ocr:
        ex.body = markdown.append_scanned_ocr(
            ex.body, scanned_ocr, total_pages=len(pages))

    # 7) Markdown 직렬화
    return markdown.to_markdown(ex)
