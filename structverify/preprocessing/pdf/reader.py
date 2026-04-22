"""
preprocessing/pdf/reader.py — PyMuPDF & Docling 기반 1차 추출

* PyMuPDF: 페이지 텍스트 레이어 + 이미지 bbox + 고해상도 crop 렌더링
* Docling: JSON + HTML 동시 export (소스 스코어링에 사용)
"""
from __future__ import annotations
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import fitz  # PyMuPDF

from structverify.preprocessing.pdf.models import PageText
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


# ── PyMuPDF 페이지 텍스트 ────────────────────────────────────────────────

def extract_page(pdf_path: str, page_no: int) -> PageText:
    """단일 페이지 PyMuPDF 추출. 프로세스 병렬에서도 동작하도록 순수 함수."""
    with fitz.open(pdf_path) as doc:
        page = doc[page_no]
        txt = page.get_text("text") or ""
        images = page.get_images(full=True)
    return PageText(
        page_no=page_no,
        text=txt,
        needs_ocr=len(txt.strip()) < 20,
        has_images=len(images) > 0,
    )


def extract_all_pages(pdf_path: str) -> list[PageText]:
    """페이지 단위 병렬 추출. 페이지 수 2 이하는 순차(오버헤드 방지)."""
    with fitz.open(pdf_path) as doc:
        n = doc.page_count
    if n <= 2:
        return [extract_page(pdf_path, i) for i in range(n)]
    workers = min(os.cpu_count() or 4, n)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(extract_page, pdf_path, i): i for i in range(n)}
        pages = [f.result() for f in as_completed(fut)]
    return sorted(pages, key=lambda p: p.page_no)


# ── PyMuPDF 이미지 bbox ──────────────────────────────────────────────────

def get_image_rects(pdf_path: str, page_no: int) -> list[tuple]:
    """페이지의 이미지 xref bbox 목록 (중복 xref 제거)."""
    rects: list[tuple] = []
    with fitz.open(pdf_path) as doc:
        page = doc[page_no]
        seen: set = set()
        for info in page.get_images(full=True):
            xref = info[0]
            if xref in seen:
                continue
            seen.add(xref)
            try:
                for r in page.get_image_rects(xref):
                    rects.append((float(r.x0), float(r.y0),
                                  float(r.x1), float(r.y1)))
            except Exception:
                continue
    return rects


def render_region_png(pdf_path: str, page_no: int, bbox: tuple | None = None,
                      dpi: int = 300) -> bytes:
    """페이지 일부 또는 전체를 PNG 바이트로 렌더링. OCR 입력으로 사용."""
    with fitz.open(pdf_path) as doc:
        page = doc[page_no]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, clip=fitz.Rect(*bbox)) \
            if bbox else page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


# ── Docling 구조화 추출 ──────────────────────────────────────────────────

def docling_extract(pdf_path: str) -> dict:
    """Docling 으로 JSON + HTML 동시 추출. 실패 시 비어있는 결과."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        logger.warning("docling 미설치 - 구조화 추출 스킵")
        return {"json": None, "html": None, "ok": False}
    try:
        doc = DocumentConverter().convert(pdf_path).document
        return {
            "json": doc.export_to_dict(),
            "html": doc.export_to_html(),
            "ok": True,
        }
    except Exception as e:
        logger.warning(f"docling 변환 실패: {e}")
        return {"json": None, "html": None, "ok": False}
