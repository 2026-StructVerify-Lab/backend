"""Component eval: claim detection."""
from __future__ import annotations

from typing import Any

from structverify.core.schemas import SourceType
from structverify.detection.claim_detector import detect_claims
from structverify.eval.schemas import ComponentDetectionRow
from structverify.preprocessing.sir_builder import build_sir


async def run_detection_suite(
    rows: list[ComponentDetectionRow],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        sir_doc = build_sir(row.text + "\n\n", SourceType.TEXT)
        claims = await detect_claims(sir_doc, config)
        if row.should_extract:
            predicted = any(
                row.text.strip() in (c.claim_text or "")
                or (c.claim_text or "").strip() in row.text.strip()
                for c in claims
            )
        else:
            predicted = any(
                row.text.strip()[:30] in (c.claim_text or "")
                for c in claims
            )
        correct = predicted == row.should_extract
        results.append(
            {
                "row_id": row.row_id,
                "should_extract": row.should_extract,
                "predicted_extract": predicted,
                "correct": correct,
                "detected_count": len(claims),
            }
        )
    return results
