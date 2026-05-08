"""
preprocessing/docx/reader.py — python-docx 기반 1차 파싱

* paragraphs: 헤딩 레벨 감지 ("Heading N" 스타일명 파싱)
* tables: 행·셀 순회, 병합 셀은 텍스트 합산
* core_properties: title / created 메타데이터
* docling_extract: Docling DOCX 변환 (선택적 보강)
"""
from __future__ import annotations
import re

import docx  # python-docx

from structverify.preprocessing.docx.models import (
    DocxExtracted, DocxParagraph, DocxTable,
)
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

_HEADING_RE = re.compile(r"^Heading\s+(\d+)$", re.IGNORECASE)
_LIST_STYLES = {"List Paragraph", "List Bullet", "List Number"}


def _heading_level(style_name: str) -> int | None:
    m = _HEADING_RE.match(style_name)
    return int(m.group(1)) if m else None


def _cell_text(cell) -> str:
    return " ".join(p.text.strip() for p in cell.paragraphs if p.text.strip())


def parse_docx(filepath: str) -> DocxExtracted:
    """python-docx로 DOCX 파일을 파싱해 DocxExtracted 반환."""
    doc = docx.Document(filepath)

    # 메타데이터
    props = doc.core_properties
    title = (props.title or "").strip()
    created = props.created
    date_str = created.strftime("%Y-%m-%d") if created else ""

    paragraphs: list[DocxParagraph] = []
    tables: list[DocxTable] = []

    # python-docx의 element 순서는 body XML 순서와 일치
    # Document.element.body 순회로 단락/테이블 순서 보존
    para_idx = 0
    tbl_idx = 0

    for block in doc.element.body:
        tag = block.tag.split("}")[-1]  # namespace 제거

        if tag == "p":
            # python-docx Paragraph 객체로 래핑
            para = docx.text.paragraph.Paragraph(block, doc)
            text = para.text.strip()
            if not text:
                continue
            style_name = para.style.name if para.style else "Normal"
            level = _heading_level(style_name)
            paragraphs.append(DocxParagraph(
                index=para_idx,
                text=text,
                style=style_name,
                heading_level=level,
            ))
            para_idx += 1

        elif tag == "tbl":
            tbl = docx.table.Table(block, doc)
            cell_rows = [
                [_cell_text(cell) for cell in row.cells]
                for row in tbl.rows
            ]
            if not cell_rows:
                continue
            # 첫 행을 헤더로 사용 (비어있으면 그대로 데이터)
            headers = cell_rows[0]
            data_rows = cell_rows[1:]
            tables.append(DocxTable(
                index=tbl_idx,
                headers=headers,
                rows=data_rows,
            ))
            tbl_idx += 1

    # title 폴백: 첫 번째 Heading 1
    if not title:
        for p in paragraphs:
            if p.heading_level == 1:
                title = p.text
                break

    return DocxExtracted(
        title=title,
        date=date_str,
        paragraphs=paragraphs,
        tables=tables,
        source_used="python-docx",
    )


def docling_extract(filepath: str) -> dict:
    """Docling으로 DOCX 구조화 추출. 실패 시 빈 결과."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        logger.warning("docling 미설치 - DOCX 구조화 추출 스킵")
        return {"json": None, "html": None, "ok": False}
    try:
        doc = DocumentConverter().convert(filepath).document
        return {
            "json": doc.export_to_dict(),
            "html": doc.export_to_html(),
            "ok": True,
        }
    except Exception as e:
        logger.warning(f"docling DOCX 변환 실패: {e}")
        return {"json": None, "html": None, "ok": False}