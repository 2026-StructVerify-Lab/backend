from unittest.mock import AsyncMock, patch

import pytest

from structverify.core.schemas import VerdictType, VerificationResult
from structverify.eval.outcome.runner import OutcomeEvalRunner
from structverify.eval.schemas import OutcomeCase
from structverify.eval.io import write_jsonl
from uuid import uuid4


@pytest.mark.asyncio
async def test_outcome_runner_smoke(tmp_path):
    ds = tmp_path / "structverify_outcome_v1"
    ds.mkdir(parents=True)
    case = OutcomeCase(
        case_id="t1",
        claim_text="2022년 고용률은 62.3%로 집계됐다.",
        expected_verdict="match",
        domain="employment",
    )
    write_jsonl(ds / "claims.jsonl", [case])

    cfg = {
        "dataset_id": "structverify_outcome_v1",
        "datasets_root": tmp_path,
        "runs_root": tmp_path / "runs",
        "eval": {"domain_oracle": True},
    }
    result = VerificationResult(claim_id=uuid4(), verdict=VerdictType.MATCH)

    with patch(
        "structverify.eval.outcome.runner.run_outcome_slice",
        new_callable=AsyncMock,
        return_value=([], [result]),
    ):
        runner = OutcomeEvalRunner(cfg)
        report = await runner.run(limit=1, run_id="test_run")
    assert report["outcome"]["n"] == 1
