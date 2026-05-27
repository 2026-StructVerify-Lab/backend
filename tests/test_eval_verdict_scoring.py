from structverify.eval.components.verdict_scoring import (
    verdict_aligned_match,
    verdict_strict_match,
)


def test_verdict_strict():
    assert verdict_strict_match("match", "match")
    assert not verdict_strict_match("mismatch", "unverifiable")


def test_verdict_aligned_gray_zone():
    # 20% error: gray zone accepts mismatch label + unverifiable pred
    assert verdict_aligned_match(
        "mismatch",
        "unverifiable",
        stated=125.0,
        official=100.0,
    )
    assert verdict_aligned_match(
        "mismatch",
        "mismatch",
        stated=150.0,
        official=100.0,
    )
