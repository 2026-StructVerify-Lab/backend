from structverify.eval.report import summarize_outcome, summarize_outcome_nested


def test_summarize_outcome_confusion():
    preds = [
        {
            "expected_verdict": "match",
            "predicted_verdict": "match",
            "verdict_correct": True,
            "value_within_tolerance": True,
            "stat_id_match": True,
        },
        {
            "expected_verdict": "match",
            "predicted_verdict": "unverifiable",
            "verdict_correct": False,
            "value_within_tolerance": True,
            "value_ok_verdict_wrong": True,
            "stat_id_match": False,
        },
    ]
    s = summarize_outcome(preds)
    assert s["n"] == 2
    assert s["verdict_accuracy"] == 0.5
    assert s["value_ok_verdict_wrong_rate"] == 0.5
    assert "match" in s["confusion"]


def test_summarize_outcome_nested():
    preds = [
        {
            "schema_mode": "oracle",
            "expected_verdict": "match",
            "predicted_verdict": "match",
            "verdict_correct": True,
            "value_within_tolerance": True,
        },
        {
            "schema_mode": "induce",
            "expected_verdict": "match",
            "predicted_verdict": "unverifiable",
            "verdict_correct": False,
            "value_within_tolerance": True,
        },
    ]
    nested = summarize_outcome_nested(preds, primary_schema_mode="oracle")
    assert nested["primary_schema_mode"] == "oracle"
    assert nested["oracle"]["verdict_accuracy"] == 1.0
    assert nested["induce"]["verdict_accuracy"] == 0.0
    assert nested["verdict_accuracy"] == 1.0
