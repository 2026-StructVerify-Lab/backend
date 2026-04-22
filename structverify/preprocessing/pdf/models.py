"""
preprocessing/pdf/models.py — PDF 파이프라인 내부용 데이터 구조

* 파이프라인 단계 사이 전달용이며, 외부 스키마(`core.schemas`)와는 분리.
* SIR Tree 로의 변환은 상위 단계(`sir_builder.py`)에서 수행.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class PageText:
    """PyMuPDF 로 추출한 페이지 단위 정보"""
    page_no: int
    text: str
    needs_ocr: bool          # 텍스트 레이어가 비어있음 → 스캔 페이지
    has_images: bool         # 이미지 블록 존재 → 이미지 OCR 대상


@dataclass
class Extracted:
    """선택된 소스(JSON/HTML)에서 뽑은 필드"""
    title: str = ""
    date: str = ""
    body: str = ""
    source_used: str = ""    # "json" | "html" | "pymupdf"


@dataclass
class ImageOcrHit:
    """이미지 블록 OCR 결과 단위"""
    idx: int                 # 페이지 내 이미지 순번
    bbox: tuple              # (x0, y0, x1, y1)
    text: str
