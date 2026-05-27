"""Run audit axis on outcome predictions + optional live results."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from structverify.core.schemas import VerdictType, VerificationResult, Evidence
from structverify.eval.audit.constraints import audit_result_row
from structverify.eval.audit.kosis_grounding import grounding_from_config
from structverify.eval.report import summarize_audit
from structverify.eval.schemas import OutcomeCase
from uuid import uuid4


def _result_from_prediction(pred: dict[str, Any]) -> VerificationResult | None:
    pv = pred.get("predicted_verdict")
    if not pv:
        return None
    ev = None
    stat = pred.get("predicted_stat_id")
    oval = pred.get("predicted_official_value")
    if stat or oval is not None:
        ev = Evidence(
            source_name="KOSIS",
            stat_table_id=stat,
            official_value=oval,
        )
    return VerificationResult(
        claim_id=uuid4(),
        verdict=VerdictType(pv),
        evidence=ev,
    )


async def run_audit_on_predictions(
    predictions: list[dict[str, Any]],
    cases_by_id: dict[str, OutcomeCase],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eval_cfg = config.get("eval") or {}
    rel = float(eval_cfg.get("value_tolerance_relative", 0.005))
    abs_tol = float(eval_cfg.get("value_tolerance_absolute", 0.1))

    rows: list[dict[str, Any]] = []
    for pred in predictions:
        case_id = pred["case_id"]
        case = cases_by_id.get(case_id)
        result = _result_from_prediction(pred)
        row = audit_result_row(
            case_id,
            result,
            stated_value=case.stated_value if case else None,
            official_value=case.official_value if case else None,
            rel=rel,
            abs_tol=abs_tol,
        )
        if result is not None:
            org_hint = case.kosis_org_id if case else None
            g = await grounding_from_config(
                result, config, org_id_hint=org_hint
            )
            row.update(g)
            row["kosis_grounding_ok"] = g.get("kosis_grounding_ok")
        rows.append(row)

    return rows, summarize_audit(rows)
