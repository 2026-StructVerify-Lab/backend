"""Component eval: schema induction."""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from structverify.core.schemas import Claim, SourceOffset, SourceType
from structverify.detection.schema_inductor import induce_schemas
from structverify.eval.components.schema_scoring import (
    indicators_match,
    schema_field_score,
    schema_values_match,
    time_periods_match,
)
from structverify.eval.schemas import ComponentSchemaRow
from structverify.preprocessing.sir_builder import build_sir


async def run_schema_suite(
    rows: list[ComponentSchemaRow],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    cfg = dict(config)
    results: list[dict[str, Any]] = []
    for row in rows:
        text = row.claim_text
        if row.context_text:
            text = f"{row.context_text}\n\n{row.claim_text}"
        sir_doc = build_sir(text, SourceType.TEXT)
        cfg["detected_domain"] = row.domain
        claim = Claim(
            claim_id=uuid4(),
            doc_id=sir_doc.doc_id,
            block_id="b0000",
            sent_id="b0000s0000",
            claim_text=row.claim_text,
            source_offset=SourceOffset(),
            context_text=row.context_text or row.claim_text,
        )
        try:
            expanded = await induce_schemas([claim], cfg, graph=None)
            sch = expanded[0].schema if expanded else None
            ind_ok = bool(
                sch
                and row.expected_indicator
                and indicators_match(row.expected_indicator, sch.indicator or "")
            )
            val_ok = True
            if row.expected_value is not None and sch and sch.value is not None:
                val_ok = schema_values_match(
                    row.expected_value,
                    sch.value,
                    expected_unit=row.expected_unit,
                    actual_unit=sch.unit,
                )
            time_ok = not row.expected_time_period or (
                sch is not None
                and time_periods_match(row.expected_time_period, sch.time_period)
            )
            field_score = schema_field_score(
                indicator_ok=ind_ok,
                value_ok=val_ok,
                time_ok=time_ok,
            )
            has_schema = bool(sch)
            correct_strict = has_schema and ind_ok and val_ok and time_ok
            correct_aligned = has_schema and field_score >= (2.0 / 3.0)
            results.append(
                {
                    "row_id": row.row_id,
                    "correct": correct_strict,
                    "correct_aligned": correct_aligned,
                    "correct_strict": correct_strict,
                    "indicator_ok": ind_ok,
                    "value_ok": val_ok,
                    "time_ok": time_ok,
                    "field_score": field_score,
                    "indicator": sch.indicator if sch else None,
                    "value": sch.value if sch else None,
                    "time_period": sch.time_period if sch else None,
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
