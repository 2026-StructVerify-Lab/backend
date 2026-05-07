"""
preprocessing/docx/models.py — DOCX 파이프라인 내부용 데이터 구조

파이프라인 단계 사이 전달용. 외부 스키마(core.schemas)와 분리.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class DocxParagraph:
    """단락 단위 정보"""
    index: int
    text: str
    style: str          # 원본 스타일명 ("Heading 1", "Normal", "List Paragraph" …)
    heading_level: int | None   # 1~9, None이면 일반 단락


@dataclass
class DocxTable:
    """테이블 단위 정보"""
    index: int
    headers: list[str]
    rows: list[list[str]]


@dataclass
class DocxExtracted:
    """파싱된 필드 집합"""
    title: str = ""
    date: str = ""
    paragraphs: list[DocxParagraph] = field(default_factory=list)
    tables: list[DocxTable] = field(default_factory=list)
    source_used: str = ""   # "python-docx" | "docling"