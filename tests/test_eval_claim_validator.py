from structverify.eval.build.claim_validator import validate_outcome_case
from structverify.eval.schemas import OutcomeCase


def test_validator_match_ok():
    case = OutcomeCase(
        case_id="t1",
        claim_text="2022년 고용률은 62.3%로 집계됐다.",
        expected_verdict="match",
        indicator="고용률",
        time_period="2022",
        unit="%",
        stated_value=62.3,
        official_value=62.3,
        reference_stat_id="DT_200Y108",
    )
    assert validate_outcome_case(case).ok is True


def test_validator_rejects_scientific_notation():
    case = OutcomeCase(
        case_id="t2",
        claim_text="2022년 GDP는 1.8654e+06십억원이다.",
        expected_verdict="match",
        indicator="GDP",
        time_period="2022",
        unit="십억원",
        stated_value=1865404.9,
        official_value=1865404.9,
        reference_stat_id="DT_X",
    )
    result = validate_outcome_case(case)
    assert result.ok is False
    assert any("scientific" in e for e in result.errors)


def test_validator_mismatch_needs_large_error():
    case = OutcomeCase(
        case_id="t3",
        claim_text="2022년 고용률은 80.0%로 집계됐다.",
        expected_verdict="mismatch",
        label_method="value_perturbation",
        indicator="고용률",
        time_period="2022",
        unit="%",
        stated_value=80.0,
        official_value=62.3,
        reference_stat_id="DT_200Y108",
    )
    result = validate_outcome_case(case)
    assert result.ok is False
