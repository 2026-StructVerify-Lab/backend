"""
preprocessing/docx — DOCX → Markdown 추출 파이프라인

책임 분리:
  models   : 데이터 구조 (DocxParagraph, DocxTable, DocxExtracted)
  reader   : python-docx 파싱 + Docling DOCX 보강
  markdown : Markdown 직렬화
  pipeline : 오케스트레이터 — `extract_docx_to_markdown`
"""
from structverify.preprocessing.docx.pipeline import extract_docx_to_markdown

__all__ = ["extract_docx_to_markdown"]