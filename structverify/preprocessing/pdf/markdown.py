"""
preprocessing/pdf/markdown.py — Markdown 직렬화 + OCR 인라인 병합

* `to_markdown`                    : Extracted → 최종 MD 문자열
* `table_to_md`                    : BeautifulSoup `<table>` → MD 표
* `inline_image_ocr_into_body`     : 이미지 OCR 결과를 본문 페이지 앵커 뒤에 삽입
* `append_scanned_ocr`             : 스캔 페이지 OCR 을 body 에 병합 (교체/추가)
"""
from __future__ import annotations

from structverify.preprocessing.pdf.models import (
    Extracted, ImageOcrHit, PageText)


def table_to_md(tbl) -> str:
    """BeautifulSoup `<table>` 엘리먼트를 Markdown 표로 변환."""
    rows: list[list[str]] = []
    for tr in tbl.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    md = ["| " + " | ".join(rows[0]) + " |",
          "| " + " | ".join(["---"] * width) + " |"]
    for r in rows[1:]:
        md.append("| " + " | ".join(r) + " |")
    return "\n".join(md)


def inline_image_ocr_into_body(body: str,
                               image_ocr: dict[int, list[ImageOcrHit]],
                               pages: list[PageText]) -> str:
    """
    본문에 이미지 OCR 결과 삽입.
    전략:
      1) 각 페이지의 PyMuPDF 텍스트 첫 유의미한 라인(10자+) 을 앵커로 검색
      2) 앵커 직후 단락 경계에 `> **[이미지 p.N-k]** <OCR>` 인용 블록 삽입
      3) 앵커 매칭 실패 시 body 말미 `## 이미지 내 텍스트` 섹션으로 폴백
    """
    if not image_ocr:
        return body

    def block(pno: int) -> str:
        lines = []
        for hit in image_ocr[pno]:
            snippet = hit.text.strip()
            if snippet:
                quoted = snippet.replace("\n", "\n> ")
                lines.append(
                    f"\n> **[이미지 p.{pno+1}-{hit.idx+1}]**\n>\n> {quoted}"
                )
        return "\n".join(lines)

    merged = body
    orphans: list[tuple[int, str]] = []
    for pno in sorted(image_ocr):
        page_text = next((p.text for p in pages if p.page_no == pno), "").strip()
        anchor = next(
            (l.strip()[:30] for l in page_text.splitlines() if len(l.strip()) >= 10),
            "",
        )
        if anchor and anchor in merged:
            idx = merged.find(anchor) + len(anchor)
            nl = merged.find("\n\n", idx)
            insert_at = nl if nl != -1 else len(merged)
            merged = merged[:insert_at] + "\n\n" + block(pno) + merged[insert_at:]
        else:
            orphans.append((pno, block(pno)))

    if orphans:
        merged += "\n\n## 이미지 내 텍스트\n"
        for pno, blk in orphans:
            merged += f"\n**p.{pno+1}**\n{blk}\n"
    return merged


def append_scanned_ocr(body: str, scanned_ocr: dict[int, str],
                       total_pages: int) -> str:
    """
    스캔 페이지 OCR 병합.
    * 전체 페이지의 50%↑ 가 스캔이면 body 전체를 OCR 결과로 교체
      (텍스트 레이어가 없는 문서이므로 Docling body 는 신뢰 불가)
    * 미만이면 말미 `## 스캔 페이지 OCR` 섹션으로 append
    """
    if not scanned_ocr:
        return body
    if len(scanned_ocr) * 2 >= max(total_pages, 1):
        return "\n\n".join(scanned_ocr[k] for k in sorted(scanned_ocr))
    out = body + "\n\n## 스캔 페이지 OCR\n"
    for pno in sorted(scanned_ocr):
        out += f"\n**p.{pno+1}**\n\n{scanned_ocr[pno]}\n"
    return out


def to_markdown(ex: Extracted) -> str:
    """Extracted → 최종 Markdown 문자열."""
    parts: list[str] = []
    if ex.title:
        parts.append(f"# {ex.title.strip()}")
    if ex.date:
        parts.append(f"_작성일: {ex.date.strip()}_")
    parts.append("")
    parts.append(ex.body.strip())
    return "\n\n".join(parts).strip() + "\n"
