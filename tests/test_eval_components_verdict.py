import pytest

from structverify.eval.components.verdict_runner import run_verdict_suite
from structverify.eval.schemas import ComponentVerdictRow


@pytest.mark.asyncio
async def test_verdict_suite_match():
    rows = [
        ComponentVerdictRow(
            row_id="v1",
            claim_text="2022년 고용률은 62.3%였다.",
            stated_value=62.3,
            official_value=62.3,
            unit="%",
            time_period="2022",
            expected_verdict="match",
        )
    ]
    results = await run_verdict_suite(rows, {})
    assert results[0]["correct"] is True
