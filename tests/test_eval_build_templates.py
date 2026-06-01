from structverify.eval.build.claim_templates import (
    build_match_claim_text,
    float_to_claim_str,
    perturb_stated_value,
)


def test_build_match_claim_text():
    t = build_match_claim_text(
        indicator="고용률",
        time_period="2022",
        value=62.3,
        unit="%",
    )
    assert "62.3" in t
    assert "고용률" in t


def test_perturb_mismatch():
    bad = perturb_stated_value(100.0, "%")
    assert bad != 100.0
    # ×1.5 → ~33% relative error (MISMATCH band, not 20% gray zone)
    assert abs(bad - 100.0) / max(bad, 100.0) > 0.30


def test_perturb_absolute():
    bad = perturb_stated_value(1000.0, "명")
    assert abs(bad - 1000.0) / bad > 0.30


def test_float_to_claim_str_no_scientific():
    s = float_to_claim_str(1865404.9)
    assert "e" not in s.lower()
    assert "E" not in s


def test_large_value_claim_text():
    t = build_match_claim_text(
        indicator="GDP",
        time_period="2022",
        value=1865404.9,
        unit="십억원",
    )
    assert "e+" not in t.lower()
    assert "e-" not in t.lower()


def test_compound_unit_claim():
    t = build_match_claim_text(
        indicator="에너지소비",
        time_period="2021",
        value=455970.0,
        unit="천TOE",
    )
    assert "천TOE" in t or "TOE" in t
    assert "e" not in t.lower()
