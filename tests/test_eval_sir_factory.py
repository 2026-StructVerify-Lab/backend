from structverify.eval.outcome.sir_factory import build_sir_for_case, claims_from_case
from structverify.eval.schemas import OutcomeCase


def test_sir_factory_aligns_claim():
    case = OutcomeCase(
        case_id="t1",
        claim_text="2022년 고용률은 62.3%로 집계됐다.",
        expected_verdict="match",
    )
    sir = build_sir_for_case(case)
    claims = claims_from_case(case, sir, schema_mode="induce")
    assert len(claims) == 1
    assert claims[0].claim_text == case.claim_text
    assert claims[0].block_id.startswith("b")
    assert claims[0].schema is None


def test_sir_factory_oracle_schema():
    case = OutcomeCase(
        case_id="t2",
        claim_text="2022년 고용률은 62.3%로 집계됐다.",
        expected_verdict="match",
        indicator="고용률",
        time_period="2022",
        unit="%",
        stated_value=62.3,
    )
    sir = build_sir_for_case(case)
    assert sir.raw_text
    claims = claims_from_case(case, sir, schema_mode="oracle")
    assert claims[0].schema is not None
    assert claims[0].schema.indicator == "고용률"
    assert claims[0].schema.value == 62.3
    assert claims[0].schema.population == "전체"
