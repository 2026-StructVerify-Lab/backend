"""
preprocessing/pdf — PDF → Markdown 추출 파이프라인

책임 분리:
  models    : 데이터 구조 (PageText, Extracted)
  reader    : PyMuPDF 텍스트/이미지 읽기 + Docling 구조화 추출
  scoring   : JSON vs HTML 소스 품질 스코어링
  fields    : 선택 소스에서 title / date / body 추출 + 정규식 폴백
  ocr       : OCR 백엔드 라우팅 (easyocr / tesseract / paddleocr)
              + 이미지 블록 crop OCR + 스캔 페이지 전체 OCR
  markdown  : 최종 Markdown 직렬화 + 표 변환 + 이미지 OCR 인라인 삽입
  pipeline  : 위 모듈들을 엮는 오케스트레이터 — `extract_pdf_to_markdown`
"""
from structverify.preprocessing.pdf.pipeline import extract_pdf_to_markdown

__all__ = ["extract_pdf_to_markdown"]
