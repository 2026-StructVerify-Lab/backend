"""tests/test_detection_candidate_heuristic.py — candidate heuristic unit tests."""
from structverify.detection.candidate.heuristic import _score_candidate_heuristic


def test_heuristic_scores_numeric_sentence():
    score, label, source, signals = _score_candidate_heuristic(
        "2024년 출생아 수는 2만 171명으로 전년 대비 감소했다.",
        threshold=0.5,
    )
    assert source == "heuristic_fallback"
    assert score >= 0.5
    assert label is True
    assert signals["has_quantity"] is True
