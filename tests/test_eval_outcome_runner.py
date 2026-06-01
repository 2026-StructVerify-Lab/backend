from unittest.mock import AsyncMock, patch

import pytest

from structverify.core.schemas import VerdictType, VerificationResult
from structverify.eval.outcome.runner import OutcomeEvalRunner
from structverify.eval.schemas import OutcomeCase, OutcomeManifest
from structverify.eval.io import write_json, write_jsonl
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
        "eval": {
            "domain_oracle": True,
            "schema_modes": ["oracle", "induce"],
            "primary_schema_mode": "oracle",
            "workspace_scope": "per_case",
        },
    }
    result = VerificationResult(claim_id=uuid4(), verdict=VerdictType.MATCH)

    with patch(
        "structverify.eval.outcome.runner.run_outcome_slice",
        new_callable=AsyncMock,
        return_value=([], [result]),
    ) as mock_slice:
        runner = OutcomeEvalRunner(cfg)
        report = await runner.run(limit=1, run_id="test_run")
    assert report["outcome"]["oracle"]["n"] == 1
    assert report["outcome"]["induce"]["n"] == 1
    assert mock_slice.call_count == 2
    assert mock_slice.call_args_list[0].kwargs.get("case_id") == "t1"


@pytest.mark.asyncio
async def test_outcome_runner_split_holdout(tmp_path):
    ds = tmp_path / "structverify_outcome_v2"
    ds.mkdir(parents=True)
    cases = [
        OutcomeCase(case_id="train_1", claim_text="a", expected_verdict="match"),
        OutcomeCase(case_id="holdout_1", claim_text="b", expected_verdict="match"),
    ]
    write_jsonl(ds / "claims.jsonl", cases)
    write_json(
        ds / "manifest.json",
        OutcomeManifest(
            dataset_id="structverify_outcome_v2",
            case_count=2,
            holdout_case_ids=["holdout_1"],
        ).model_dump(),
    )

    cfg = {
        "dataset_id": "structverify_outcome_v2",
        "datasets_root": tmp_path,
        "runs_root": tmp_path / "runs",
        "eval": {"schema_modes": ["oracle"]},
    }
    result = VerificationResult(claim_id=uuid4(), verdict=VerdictType.MATCH)

    with patch(
        "structverify.eval.outcome.runner.run_outcome_slice",
        new_callable=AsyncMock,
        return_value=([], [result]),
    ):
        runner = OutcomeEvalRunner(cfg, split="train")
        report = await runner.run(run_id="split_run")
    assert report["outcome"]["oracle"]["n"] == 1
