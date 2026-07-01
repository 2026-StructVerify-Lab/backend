"""detection/domain/preview.py — SIR 문서 미리보기 텍스트.

domain_classifier.py에서 분리 (로직 move-only).

[김예슬 - 2026-04-22] _build_text_preview — 블록 타입 고려
"""
from __future__ import annotations

from structverify.core.schemas import SIRDocument


def _build_text_preview(sir_doc: SIRDocument, max_chars: int = 600) -> str:
    """
    SIR 문서에서 분류에 유용한 미리보기 텍스트를 구성한다.
    heading 블록 우선, 이후 paragraph 추가. table/list 제외.
    """
    from structverify.core.schemas import BlockType

    parts: list[str] = []
    total = 0

    for block in sir_doc.blocks:
        if block.type == BlockType.HEADING and block.content:
            parts.append(block.content.strip())
            total += len(block.content)
            if total >= max_chars:
                break

    for block in sir_doc.blocks:
        if block.type == BlockType.PARAGRAPH and block.content:
            parts.append(block.content.strip())
            total += len(block.content)
            if total >= max_chars:
                break

    return " ".join(parts)[:max_chars]
