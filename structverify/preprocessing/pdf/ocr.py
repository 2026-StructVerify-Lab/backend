"""
preprocessing/pdf/ocr.py — OCR 백엔드 라우팅 + 이미지/스캔 OCR

백엔드 비교 (한국어 기준 벤치마크):
  * EasyOCR    : 한글 정확도 가장 우수, 초기 모델 로드 느림 → 싱글톤
  * Tesseract  : 빠르지만 한글 정확도 약함 (영문 위주 문서에만 추천)
  * PaddleOCR  : 정확도·속도 절충, 한국어 모델 별도 설치 필요

환경변수 `OCR_BACKEND` (기본: easyocr) 로 교체.
실제 라이브러리는 지연 로드 — 설치돼 있지 않으면 해당 백엔드만 실패하고
다른 백엔드/파이프라인 단계는 정상 동작.
"""
from __future__ import annotations
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

from structverify.preprocessing.pdf.models import PageText, ImageOcrHit
from structverify.preprocessing.pdf.reader import (
    get_image_rects, render_region_png)
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_BACKEND = "easyocr"

# 리더 싱글톤 (모델 재로딩 비용 방지)
_easy_reader = None
_paddle_reader = None


# ── 백엔드 함수 ─────────────────────────────────────────────────────────

def _ocr_easyocr(img) -> str:
    global _easy_reader
    import easyocr  # type: ignore
    import numpy as np
    if _easy_reader is None:
        _easy_reader = easyocr.Reader(["ko", "en"], gpu=False)
    return "\n".join(_easy_reader.readtext(np.array(img), detail=0))


def _ocr_tesseract(img) -> str:
    import pytesseract  # type: ignore
    return pytesseract.image_to_string(img, lang="kor+eng")


def _ocr_paddle(img) -> str:
    global _paddle_reader
    from paddleocr import PaddleOCR  # type: ignore
    import numpy as np
    if _paddle_reader is None:
        _paddle_reader = PaddleOCR(
            use_angle_cls=True, lang="korean", show_log=False)
    res = _paddle_reader.ocr(np.array(img), cls=True)
    out: list[str] = []
    for block in res or []:
        for line in block or []:
            if line and len(line) >= 2 and line[1]:
                out.append(line[1][0])
    return "\n".join(out)


_BACKENDS = {
    "easyocr": _ocr_easyocr,
    "tesseract": _ocr_tesseract,
    "paddleocr": _ocr_paddle,
}


def resolve_backend(backend: str | None = None) -> str:
    b = backend or os.environ.get("OCR_BACKEND", DEFAULT_BACKEND)
    if b not in _BACKENDS:
        raise ValueError(f"unknown OCR backend: {b}")
    return b


def run_ocr(img, backend: str | None = None) -> str:
    return _BACKENDS[resolve_backend(backend)](img)


def benchmark(img, backends=("tesseract", "easyocr")) -> dict:
    """개발용 벤치 — 백엔드별 처리시간/글자수/샘플 비교."""
    import time
    report: dict = {}
    for name in backends:
        fn = _BACKENDS[name]
        t0 = time.time()
        try:
            out = fn(img)
            report[name] = {
                "ok": True,
                "elapsed": round(time.time() - t0, 2),
                "chars": len(out),
                "sample": out[:160],
            }
        except Exception as e:
            report[name] = {"ok": False,
                            "error": f"{type(e).__name__}: {e}"}
    return report


# ── 이미지 블록 crop OCR ────────────────────────────────────────────────

def _open_png(png_bytes: bytes):
    if Image is None:
        raise RuntimeError("Pillow 미설치 - OCR 불가")
    return Image.open(io.BytesIO(png_bytes))


def ocr_page_images(pdf_path: str, page_no: int,
                    backend: str | None = None) -> list[ImageOcrHit]:
    """페이지 내 이미지 블록 각각에 대해 crop → OCR."""
    hits: list[ImageOcrHit] = []
    for i, bb in enumerate(get_image_rects(pdf_path, page_no)):
        # 아이콘/로고(30pt 미만) 는 OCR 가치 낮음 → skip
        if (bb[2] - bb[0] < 30) or (bb[3] - bb[1] < 30):
            continue
        try:
            png = render_region_png(pdf_path, page_no, bb, dpi=300)
            text = run_ocr(_open_png(png), backend).strip()
        except Exception as e:
            logger.warning(f"image OCR 실패 p{page_no} #{i}: {e}")
            continue
        if text:
            hits.append(ImageOcrHit(idx=i, bbox=bb, text=text))
    return hits


def collect_image_ocr(pdf_path: str, pages: list[PageText],
                      backend: str | None = None,
                      max_workers: int = 4) -> dict[int, list[ImageOcrHit]]:
    """이미지 포함 페이지만 Thread 병렬. 결과 {page_no: [ImageOcrHit...]}"""
    targets = [p.page_no for p in pages if p.has_images]
    if not targets:
        return {}
    out: dict[int, list[ImageOcrHit]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut = {ex.submit(ocr_page_images, pdf_path, pno, backend): pno
               for pno in targets}
        for f in as_completed(fut):
            res = f.result()
            if res:
                out[fut[f]] = res
    return out


# ── 스캔 페이지 전체 OCR ────────────────────────────────────────────────

def collect_scanned_ocr(pdf_path: str, pages: list[PageText],
                        backend: str | None = None,
                        max_workers: int = 4) -> dict[int, str]:
    """`needs_ocr=True` 페이지를 full-page 렌더 → OCR."""
    targets = [p.page_no for p in pages if p.needs_ocr]
    if not targets or Image is None:
        return {}

    def _full(pno: int) -> str:
        png = render_region_png(pdf_path, pno, bbox=None, dpi=220)
        return run_ocr(_open_png(png), backend)

    out: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut = {ex.submit(_full, pno): pno for pno in targets}
        for f in as_completed(fut):
            out[fut[f]] = f.result()
    return out
