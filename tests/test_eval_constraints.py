from uuid import uuid4

from structverify.core.schemas import Evidence, VerdictType, VerificationResult
from structverify.eval.audit.constraints import check_constraints


def test_r1_no_evidence():
    r = VerificationResult(claim_id=uuid4(), verdict=VerdictType.MATCH, evidence=None)
    v = check_constraints(r)
    assert any(x["rule_id"] == "R1" for x in v)


def test_r2_value_without_stat_id():
    r = VerificationResult(
        claim_id=uuid4(),
        verdict=VerdictType.MATCH,
        evidence=Evidence(source_name="KOSIS", official_value=1.0, stat_table_id=None),
    )
    v = check_constraints(r)
    assert any(x["rule_id"] == "R2" for x in v)
