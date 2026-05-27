"""Component eval: deterministic verify_claim with mock evidence."""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from structverify.core.schemas import Claim, ClaimSchema, Evidence, SourceOffset
from structverify.eval.components.verdict_scoring import (
    verdict_aligned_match,
    verdict_strict_match,
)
from structverify.eval.schemas import ComponentVerdictRow
from structverify.verification.verifier import verify_claim


async def run_verdict_suite(
    rows: list[ComponentVerdictRow],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        claim = Claim(
            claim_id=uuid4(),
            doc_id=uuid4(),
            block_id="b0000",
            sent_id="b0000s0000",
            claim_text=row.claim_text,
            schema=ClaimSchema(
                value=row.stated_value,
                unit=row.unit,
                time_period=row.time_period,
                parent_path=None,
                modifier=None,
            ),
            source_offset=SourceOffset(),
        )
        evidence = Evidence(
            source_name="KOSIS",
            stat_table_id="eval_mock",
            official_value=row.official_value,
            unit=row.unit,
            time_period=row.time_period,
        )
        try:
            result = verify_claim(claim, evidence, config)
            predicted = result.verdict.value
            correct_strict = verdict_strict_match(row.expected_verdict, predicted)
            correct_aligned = verdict_aligned_match(
                row.expected_verdict,
                predicted,
                stated=row.stated_value,
                official=row.official_value,
            )
            results.append(
                {
                    "row_id": row.row_id,
                    "correct": correct_aligned,
                    "correct_strict": correct_strict,
                    "predicted_verdict": predicted,
                    "expected_verdict": row.expected_verdict,
                }
            )
        except Exception as e:
            results.append(
                {
                    "row_id": row.row_id,
                    "correct": False,
                    "correct_strict": False,
                    "error": str(e),
                }
            )
    return results
