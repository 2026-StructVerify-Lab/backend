"""tests/test_detection_expand_temporal.py — expand·temporal_hints unit tests (API 키 불필요)."""
from uuid import uuid4

import pytest

from structverify.core.schemas import Claim, ClaimSchema
from structverify.detection.schema.expand import (
    _dedup_null_schemas,
    _expand_claims_from_schemas,
)
from structverify.detection.schema.temporal_hints import (
    _build_temporal_hint,
    _count_temporal_in_text,
)


def _schema(**kwargs) -> ClaimSchema:
    defaults = {"parent_path": None, "modifier": None}
    defaults.update(kwargs)
    return ClaimSchema.model_construct(**defaults)


def _claim(**kwargs) -> Claim:
    defaults = {
        "doc_id": uuid4(),
        "block_id": "b1",
        "sent_id": "s1",
        "claim_text": "test claim",
    }
    defaults.update(kwargs)
    return Claim.model_construct(**defaults)


class TestDedupNullSchemas:
    def test_drops_null_after_value_schema_same_key(self):
        schemas = [
            _schema(indicator="합계출산율", time_period="2024", value=0.79),
            _schema(indicator="합계출산율", time_period="2024", value=None),
        ]
        assert _dedup_null_schemas(schemas) == [schemas[0]]

    def test_keeps_null_when_first_for_key(self):
        schemas = [
            _schema(indicator="합계출산율", time_period="2024", value=None),
            _schema(indicator="합계출산율", time_period="2024", value=0.79),
        ]
        assert len(_dedup_null_schemas(schemas)) == 2

    def test_different_population_not_merged(self):
        schemas = [
            _schema(
                indicator="출산율", time_period="2024",
                population="동작구", value=10.6,
            ),
            _schema(
                indicator="출산율", time_period="2024",
                population="성동구", value=8.9,
            ),
        ]
        assert len(_dedup_null_schemas(schemas)) == 2

    def test_empty_list(self):
        assert _dedup_null_schemas([]) == []


class TestExpandClaimsFromSchemas:
    def test_single_schema_attaches_to_original_claim(self):
        claim = _claim(sent_id="s42")
        schema = _schema(indicator="출생아 수", value=238000.0)
        out = _expand_claims_from_schemas(claim, [schema])
        assert len(out) == 1
        assert out[0] is claim
        assert claim.schema is schema

    def test_multiple_schemas_clone_with_new_claim_ids(self):
        claim = _claim(sent_id="s42")
        schemas = [
            _schema(indicator="a", value=1.0),
            _schema(indicator="b", value=2.0),
            _schema(indicator="c", value=3.0),
        ]
        out = _expand_claims_from_schemas(claim, schemas)
        assert len(out) == 3
        assert out[0].schema.indicator == "a"
        assert out[1].schema.indicator == "b"
        assert out[2].schema.indicator == "c"
        ids = {c.claim_id for c in out}
        assert len(ids) == 3
        assert out[1].sent_id == out[2].sent_id == "s42"

    def test_empty_schemas_returns_empty(self):
        claim = _claim()
        assert _expand_claims_from_schemas(claim, []) == []


class TestCountTemporalInText:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("", 0),
            ("작년 X도, 재작년 Y도", 2),
            ("재작년만 언급", 1),
            ("올해와 내년 전망", 2),
        ],
    )
    def test_counts(self, text, expected):
        assert _count_temporal_in_text(text) == expected

    def test_rejaeon_not_double_counts_jaknyeon(self):
        assert _count_temporal_in_text("재작년 실적") == 1


class _FakeGraph:
    def __init__(
        self,
        *,
        prov=None,
        anchor_year=None,
        te_count=0,
    ):
        self._prov = prov
        self._anchor_year = anchor_year
        self._te_count = te_count

    def temporal_provenance(self, claim):
        return self._prov

    def get_anchor_year(self):
        return self._anchor_year

    def count_temporal_expressions(self, claim):
        return self._te_count


class TestBuildTemporalHint:
    def test_resolved_single_temporal(self):
        claim = _claim(claim_text="작년 출생아 수는 감소했다.")
        graph = _FakeGraph(
            prov={
                "expression": "작년",
                "resolved": "2024",
                "basis": "anchor",
            },
            anchor_year=2025,
            te_count=1,
        )
        hint = _build_temporal_hint(graph, claim)
        assert "그래프 해소 결과" in hint
        assert "2024" in hint
        assert "둘 이상" not in hint

    def test_multi_temporal_uses_anchor_table_not_prov(self):
        claim = _claim(claim_text="작년 X도, 재작년 Y도")
        graph = _FakeGraph(
            prov={
                "expression": "작년",
                "resolved": "2024",
                "basis": "anchor",
            },
            anchor_year=2025,
            te_count=2,
        )
        hint = _build_temporal_hint(graph, claim)
        assert "문서 anchor" in hint
        assert "둘 이상" in hint
        assert "그래프 해소 결과" not in hint

    def test_text_scan_triggers_multi_when_graph_undercounts(self):
        claim = _claim(claim_text="작년 X도, 재작년 Y도")
        graph = _FakeGraph(
            prov={
                "expression": "작년",
                "resolved": "2024",
            },
            anchor_year=2025,
            te_count=1,
        )
        hint = _build_temporal_hint(graph, claim)
        assert "둘 이상" in hint
        assert "그래프 해소 결과" not in hint

    def test_anchor_conversion_table(self):
        claim = _claim(claim_text="내년 전망")
        graph = _FakeGraph(prov=None, anchor_year=2025, te_count=1)
        hint = _build_temporal_hint(graph, claim)
        assert "anchor_year): 2025" in hint
        assert "→ 2026" in hint
        assert "→ 2024" in hint

    def test_no_anchor_no_prov_returns_empty(self):
        claim = _claim()
        graph = _FakeGraph(prov=None, anchor_year=None, te_count=0)
        assert _build_temporal_hint(graph, claim) == ""
