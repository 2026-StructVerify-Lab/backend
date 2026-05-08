"""
preprocessing/docx/markdown.py — DocxExtracted → Markdown 직렬화

헤딩 레벨, 리스트 스타일, 테이블을 Markdown 문법으로 변환.
테이블은 단락 흐름 사이의 삽입 위치를 유지하지 않고 문서 말미에 모은다.
(python-docx element 순서에서 테이블 idx가 단락 idx와 섞이지 않으므로)
"""
from __future__ import annotations

from structverify.preprocessing.docx.models import DocxExtracted, DocxTable

_LIST_STYLES = {"List Paragraph", "List Bullet", "List Number"}
_NUMBERED_STYLES = {"List Number"}


def _table_to_md(tbl: DocxTable) -> str:
    all_rows = [tbl.headers] + tbl.rows
    if not all_rows:
        return ""
    width = max(len(r) for r in all_rows)
    padded = [r + [""] * (width - len(r)) for r in all_rows]

    md = ["| " + " | ".join(padded[0]) + " |",
          "| " + " | ".join(["---"] * width) + " |"]
    for row in padded[1:]:
        md.append("| " + " | ".join(row) + " |")
    return "\n".join(md)


def to_markdown(ex: DocxExtracted) -> str:
    parts: list[str] = []

    if ex.title:
        parts.append(f"# {ex.title.strip()}")
    if ex.date:
        parts.append(f"_작성일: {ex.date.strip()}_")
    if parts:
        parts.append("")

    list_counter: dict[int, int] = {}  # indent_level → 번호 카운터

    for para in ex.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        if para.heading_level is not None:
            hashes = "#" * min(para.heading_level, 6)
            parts.append(f"{hashes} {text}")
            list_counter.clear()
            continue

        if para.style in _LIST_STYLES:
            if para.style in _NUMBERED_STYLES:
                cnt = list_counter.get(0, 0) + 1
                list_counter[0] = cnt
                parts.append(f"{cnt}. {text}")
            else:
                list_counter.clear()
                parts.append(f"- {text}")
            continue

        list_counter.clear()
        parts.append(text)

    # 테이블은 문서 순서대로 말미에 추가
    if ex.tables:
        parts.append("\n---\n")
        for tbl in ex.tables:
            parts.append(_table_to_md(tbl))
            parts.append("")

    return "\n\n".join(parts).strip() + "\n"