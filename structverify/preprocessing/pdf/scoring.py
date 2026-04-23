"""
preprocessing/pdf/scoring.py — JSON vs HTML 소스 품질 스코어링

Docling 이 둘 다 생성 가능하지만 문서별로 어느 쪽이 후속 필드 추출에
유리한지 다르다. 간단한 휴리스틱으로 점수를 매겨 높은 쪽을 선택.
"""
from __future__ import annotations
import json
import re
from typing import Literal

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore

from structverify.utils.logger import get_logger

logger = get_logger(__name__)


def score_json(j: dict | None) -> int:
    if not j:
        return -999
    s = 0
    flat = json.dumps(j, ensure_ascii=False)
    if re.search(r'"(title|heading)"', flat):
        s += 3
    if re.search(r'"(date|published|created)"', flat, re.I):
        s += 3
    if len(re.findall(r'"text"\s*:\s*"([^"]{2,})"', flat)) >= 20:
        s += 2
    if re.search(r'"(cells?|rows?|columns?)"', flat):
        s += 2
    return s


def score_html(h: str | None) -> int:
    if not h or BeautifulSoup is None:
        return -999
    s = 0
    soup = BeautifulSoup(h, "lxml")
    if len(soup.find_all("table")) >= 3:
        s += 2
    tags = {t.name for t in soup.find_all(
        ["h1", "h2", "h3", "p", "table", "ul", "ol"])}
    if len(tags) >= 3:
        s += 2
    if soup.find(["h1", "h2"]):
        s += 1
    if soup.find(string=re.compile(r"\d{4}[-./]\d{1,2}[-./]\d{1,2}")):
        s += 1
    return s


def pick_source(docling_result: dict) -> Literal["json", "html"]:
    """둘 다 가능할 때 높은 점수. 동점 시 JSON (구조화 메타가 풍부한 경향)."""
    sj = score_json(docling_result.get("json"))
    sh = score_html(docling_result.get("html"))
    logger.info(f"source score | json={sj} html={sh}")
    return "json" if sj >= sh else "html"
