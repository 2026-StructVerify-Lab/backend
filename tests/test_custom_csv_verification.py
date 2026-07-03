"""#69 custom_csv end-to-end verification tests.

KOSIS/API/LLM 없이 tests/fixtures/sample_custom.csv 로 agent loop 및
pipeline 경로에서 MATCH/MISMATCH verdict가 나오는지 확인한다.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

import structverify.agent.tools  # noqa: F401 — Tool registry
import structverify.retrieval.custom_csv_source  # noqa: F401
from structverify.agent.loop import LoopConfig, agent_loop
from structverify.agent.schemas import ActionType, ClaimType, Plan, PlanStep
from structverify.agent.workspace import build_workspace
from structverify.core.pipeline import VerificationPipeline, verify_text
from structverify.core.schemas import (
    Claim,
    ClaimSchema,
    SourceOffset,
    VerdictType as CoreVerdictType,
)
from structverify.retrieval.custom_csv_source import CustomCSVDataSource

pytestmark = pytest.mark.asyncio

_FIXTURE = str(Path(__file__).parent / "fixtures" / "sample_custom.csv")


def _custom_csv_config(tmp_path: Path) -> dict:
    return {
        "eval": {"bypass_detection": True},
        "agent": {
            "enabled": True,
            "workspace": {
                "backend": "local",
                "local_path": str(tmp_path / "agent_workspace"),
            },
        },
        "data_sources": {
            "enabled": ["custom_csv"],
            "default_source": "custom_csv",
            "custom_csv": {"csv_path": _FIXTURE},
            "kosis": {
                "catalog_ranker": {"enabled": False},
                "relevance_guard": {"enabled": False},
            },
        },
        "verification": {"min_confidence": 0.5},
    }


def _make_claim(
    *,
    claim_text: str,
    indicator: str,
    value: float,
    time_period: str,
    unit: str = "%",
    population: str = "전국",
) -> Claim:
    return Claim(
        doc_id=uuid4(),
        block_id="b0",
        sent_id="s0",
        claim_text=claim_text,
        schema=ClaimSchema(
            indicator=indicator,
            value=value,
            unit=unit,
            time_period=time_period,
            population=population,
            value_role="base",
            parent_path="",
            modifier=None,
        ),
        source_offset=SourceOffset(),
        check_worthy_score=1.0,
    )


def _absolute_plan(claim_id: str, *, query: str) -> Plan:
    return Plan(
        claim_id=claim_id,
        claim_type=ClaimType.ABSOLUTE,
        required_data=[],
        initial_steps=[
            PlanStep(
                action=ActionType.CATALOG_SEARCH,
                input={"query": query, "source": "custom_csv", "top_k": 5},
                rationale="custom_csv 카탈로그 검색",
            ),
            PlanStep(
                action=ActionType.FETCH_EVIDENCE,
                input={
                    "candidate_id": "<catalog_search 결과의 top id>",
                    "source": "custom_csv",
                    "params": {},
                },
                rationale="CSV 행에서 공식 수치 조회",
            ),
        ],
    )


def _datasources() -> dict[str, CustomCSVDataSource]:
    return {"custom_csv": CustomCSVDataSource(csv_path=_FIXTURE)}


async def test_agent_loop_custom_csv_match_employment_rate(tmp_path: Path) -> None:
    claim = _make_claim(
        claim_text="2023년 전국 고용률은 62.6%이다.",
        indicator="고용률",
        value=62.6,
        time_period="2023",
        unit="%",
    )
    plan = _absolute_plan(str(claim.claim_id), query="고용률")
    workspace = build_workspace(
        job_id=str(claim.doc_id),
        config=_custom_csv_config(tmp_path)["agent"]["workspace"],
    )
    workspace.initialize(source_text=claim.claim_text)
    workspace.create_claim_dir(claim.claim_id, claim_data=claim.model_dump(mode="json"))

    verdict = await agent_loop(
        plan=plan,
        claim=claim,
        workspace=workspace,
        datasources=_datasources(),
        config=_custom_csv_config(tmp_path),
        loop_config=LoopConfig(mode="deterministic", max_iterations=5),
    )

    assert verdict.verdict.value == "match"
    assert verdict.confidence >= 0.5


async def test_agent_loop_custom_csv_mismatch_employment_rate(tmp_path: Path) -> None:
    claim = _make_claim(
        claim_text="2023년 전국 고용률은 70.0%이다.",
        indicator="고용률",
        value=70.0,
        time_period="2023",
        unit="%",
    )
    plan = _absolute_plan(str(claim.claim_id), query="고용률")
    workspace = build_workspace(
        job_id=str(claim.doc_id),
        config=_custom_csv_config(tmp_path)["agent"]["workspace"],
    )
    workspace.initialize(source_text=claim.claim_text)
    workspace.create_claim_dir(claim.claim_id, claim_data=claim.model_dump(mode="json"))

    verdict = await agent_loop(
        plan=plan,
        claim=claim,
        workspace=workspace,
        datasources=_datasources(),
        config=_custom_csv_config(tmp_path),
        loop_config=LoopConfig(mode="deterministic", max_iterations=5),
    )

    assert verdict.verdict.value == "mismatch"


async def test_agent_loop_custom_csv_match_birth_count(tmp_path: Path) -> None:
    claim = _make_claim(
        claim_text="2023년 전국 출생아수는 230000명이다.",
        indicator="출생아수",
        value=230_000.0,
        time_period="2023",
        unit="명",
        population="전국",
    )
    plan = _absolute_plan(str(claim.claim_id), query="출생아수")
    workspace = build_workspace(
        job_id=str(claim.doc_id),
        config=_custom_csv_config(tmp_path)["agent"]["workspace"],
    )
    workspace.initialize(source_text=claim.claim_text)
    workspace.create_claim_dir(claim.claim_id, claim_data=claim.model_dump(mode="json"))

    verdict = await agent_loop(
        plan=plan,
        claim=claim,
        workspace=workspace,
        datasources=_datasources(),
        config=_custom_csv_config(tmp_path),
        loop_config=LoopConfig(mode="deterministic", max_iterations=5),
    )

    assert verdict.verdict.value == "match"


async def _fake_induce_schemas(claims: list[Claim], config: dict, **kwargs) -> list[Claim]:
    for claim in claims:
        if "고용률" in claim.claim_text and "70.0" in claim.claim_text:
            claim.schema = ClaimSchema(
                indicator="고용률",
                value=70.0,
                unit="%",
                time_period="2023",
                population="전국",
                value_role="base",
                parent_path="",
                modifier=None,
            )
        elif "고용률" in claim.claim_text:
            claim.schema = ClaimSchema(
                indicator="고용률",
                value=62.6,
                unit="%",
                time_period="2023",
                population="전국",
                value_role="base",
                parent_path="",
                modifier=None,
            )
    return claims


async def test_pipeline_custom_csv_only_match(tmp_path: Path) -> None:
    """VerificationPipeline + custom_csv only — KOSIS/LLM 단계는 mock."""
    cfg = _custom_csv_config(tmp_path)
    text = "2023년 전국 고용률은 62.6%이다."

    with (
        patch(
            "structverify.agent.runtime_agent.classify_domain",
            new_callable=AsyncMock,
            return_value=("labor", "노동"),
        ),
        patch(
            "structverify.agent.runtime_agent.induce_schemas",
            side_effect=_fake_induce_schemas,
        ),
        patch(
            "structverify.agent.runtime_agent.build_document_temporal_graph",
            new_callable=AsyncMock,
            return_value=([], []),
        ),
        patch(
            "structverify.agent.runtime_agent.generate_explanation",
            new_callable=AsyncMock,
            return_value="custom_csv 검증 설명",
        ),
        patch(
            "structverify.agent.planner.Planner.plan",
            new_callable=AsyncMock,
        ) as mock_plan,
        patch(
            "structverify.storage.db_manager.DBManager.save_claims",
            new_callable=AsyncMock,
        ),
        patch(
            "structverify.storage.db_manager.DBManager.save_results",
            new_callable=AsyncMock,
        ),
    ):
        async def _plan_side_effect(claim, **kwargs):
            return _absolute_plan(str(claim.claim_id), query="고용률")

        mock_plan.side_effect = _plan_side_effect

        report = await VerificationPipeline(cfg).run(text, "text")

    assert len(report.results) == 1
    assert report.results[0].verdict == CoreVerdictType.MATCH


async def test_verify_text_custom_csv_only_match(tmp_path: Path) -> None:
    cfg = _custom_csv_config(tmp_path)
    text = "2023년 전국 고용률은 62.6%이다."

    with (
        patch(
            "structverify.agent.runtime_agent.classify_domain",
            new_callable=AsyncMock,
            return_value=("labor", "노동"),
        ),
        patch(
            "structverify.agent.runtime_agent.induce_schemas",
            side_effect=_fake_induce_schemas,
        ),
        patch(
            "structverify.agent.runtime_agent.build_document_temporal_graph",
            new_callable=AsyncMock,
            return_value=([], []),
        ),
        patch(
            "structverify.agent.runtime_agent.generate_explanation",
            new_callable=AsyncMock,
            return_value="ok",
        ),
        patch(
            "structverify.agent.planner.Planner.plan",
            new_callable=AsyncMock,
        ) as mock_plan,
        patch(
            "structverify.storage.db_manager.DBManager.save_claims",
            new_callable=AsyncMock,
        ),
        patch(
            "structverify.storage.db_manager.DBManager.save_results",
            new_callable=AsyncMock,
        ),
    ):
        async def _plan_side_effect(claim, **kwargs):
            return _absolute_plan(str(claim.claim_id), query="고용률")

        mock_plan.side_effect = _plan_side_effect

        report = await verify_text(text, config=cfg)

    assert len(report.results) == 1
    assert report.results[0].verdict == CoreVerdictType.MATCH
