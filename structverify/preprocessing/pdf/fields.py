"""
preprocessing/pdf/fields.py — 선택 소스에서 제목/작성일/본문 추출

* `extract_from_json` : Docling JSON 트리를 walk 해서 라벨 기반 추출
* `extract_from_html` : BeautifulSoup 로 태그 기반 추출
* `fallback_from_plain` : PyMuPDF 평문에서 정규식 폴백 (title/date)
"""
from __future__ import annotations
import re

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore

from structverify.preprocessing.pdf.models import Extracted


DATE_RX = re.compile(
    r"(20\d{2}|19\d{2})[\s\-./년]\s*(\d{1,2})[\s\-./월]\s*(\d{1,2})"
)


def fallback_from_plain(plain: str, field_name: str) -> str:
    """Docling 이 실패/부실할 때 PyMuPDF 평문에서 필드 추출."""
    if field_name == "title":
        lines = [l.strip() for l in plain.splitlines()[:15] if l.strip()]
        return max(lines, key=len) if lines else ""
    if field_name == "date":
        m = DATE_RX.search(plain)
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        return ""
    return ""


def extract_from_json(j: dict) -> Extracted:
    title = date = ""
    body_parts: list[str] = []

    def walk(node):
        nonlocal title, date
        if isinstance(node, dict):
            t = (node.get("label") or node.get("type") or "").lower()
            txt = node.get("text") or ""
            if not title and t in ("title", "heading", "section_header") and txt:
                title = txt.strip()
            if not date and t in ("date", "published", "created") and txt:
                date = txt.strip()
            if txt and t not in ("footer", "header", "page_number"):
                body_parts.append(txt)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for x in node:
                walk(x)

    walk(j)
    return Extracted(title=title, date=date,
                     body="\n\n".join(body_parts), source_used="json")


def extract_from_html(h: str) -> Extracted:
    if BeautifulSoup is None:
        return Extracted(body=h, source_used="html-raw")

    soup = BeautifulSoup(h, "lxml")
    title_el = soup.find(["h1", "h2", "title"])
    title = title_el.get_text(strip=True) if title_el else ""

    date = ""
    meta = soup.find("meta", attrs={"name": re.compile("date|published", re.I)})
    if meta and meta.get("content"):
        date = meta["content"].strip()
    if not date and soup.find("time"):
        date = soup.find("time").get_text(strip=True)
    if not date:
        m = DATE_RX.search(soup.get_text(" "))
        if m:
            date = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # 표는 여기서 바로 MD 로 바꾸지 않고 markdown 모듈에 위임
    from structverify.preprocessing.pdf.markdown import table_to_md
    parts: list[str] = []
    for el in soup.find_all(["h1", "h2", "h3", "p", "li", "table"]):
        if el is title_el:
            continue
        if el.name == "table":
            parts.append(table_to_md(el))
        else:
            txt = el.get_text(" ", strip=True)
            if txt:
                parts.append(txt)
    return Extracted(title=title, date=date,
                     body="\n\n".join(parts), source_used="html")
