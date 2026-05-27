from structverify.eval.build.kosis_meta import is_skippable_probe_unit, resolve_kosis_org_id
from structverify.retrieval.base_connector import StatRecord


def test_resolve_kosis_org_id_from_record():
    rec = StatRecord(
        stat_id="DT_136022_10072",
        stat_name="t",
        org_name="통계청",
        org_id="136022",
    )
    assert resolve_kosis_org_id(rec) == "136022"


def test_resolve_kosis_org_id_from_stat_id():
    rec = StatRecord(stat_id="DT_136022_10072", stat_name="t", org_name="")
    assert resolve_kosis_org_id(rec) == "136022"


def test_skippable_compound_unit():
    assert is_skippable_probe_unit("명 %") is True
    assert is_skippable_probe_unit("명 % 점") is True
    assert is_skippable_probe_unit("개%") is True


def test_acceptable_units():
    assert is_skippable_probe_unit("%") is False
    assert is_skippable_probe_unit("명") is False
    assert is_skippable_probe_unit("십억원") is False
    assert is_skippable_probe_unit("천TOE") is False
