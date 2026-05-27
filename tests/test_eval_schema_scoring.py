from structverify.eval.components.schema_scoring import (
    indicators_match,
    schema_values_match,
    time_periods_match,
)


def test_indicators_token_overlap():
    assert indicators_match(
        "아버지의 교육정도별 학생 1인당 월평균 사교육비",
        "학생 1인당 월평균 사교육비",
    )


def test_time_periods_same_year():
    assert time_periods_match("202201", "2022")
    assert time_periods_match("2022", "202201")
    assert time_periods_match("2020", "2020")


def test_time_periods_mismatch():
    assert not time_periods_match("2020", "2021")


def test_schema_values_man_scale():
    assert schema_values_match(
        3157750.0,
        315.775,
        expected_unit="명",
        actual_unit="명",
    )


def test_schema_values_thousand_scale():
    assert schema_values_match(125.0, 125000.0, expected_unit="천원", actual_unit="원")
