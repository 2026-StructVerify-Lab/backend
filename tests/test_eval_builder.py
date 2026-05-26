"""Unit tests for eval set builder components."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from structverify.eval.builder.domain_mapping import (
    DOMAIN_CATEGORY_KEYWORDS,
    all_eval_domains,
    get_category_keywords,
)
from structverify.eval.builder.gold_builder import (
    _latest_year_from_stat_rec,
    _perturb_time_period,
    _perturb_value,
)
from structverify.eval.builder.kosis_sampler import KosisRowSampler
from structverify.eval.builder.schemas import (
    BuildState,
    ClaimSpec,
    EvalArticle,
    EvalClaim,
    EvalManifest,
    GoldEvidence,
    GoldSchema,
)
from structverify.eval.builder.unverifiable_recipes import allocate_verdict_targets
from structverify.eval.builder.story_coherence import (
    StoryAnchor,
    catalog_path_is_national,
    catalog_row_compatible,
    catalog_row_is_national,
    extract_region_token,
    normalize_region_token,
    path_is_local,
    region_compatible_with_anchor,
    region_token_for_spec,
    spec_compatible_with_anchor,
    is_story_lock_region,
    pick_viable_national_pool,
    pick_viable_region_pool,
    specs_single_region,
)
from structverify.eval.builder.set_builder import EvalSetBuilder
from structverify.eval.builder.spec_preflight import validate_specs_before_prose
from structverify.eval.builder.validator import EvalArticleValidator
from structverify.eval.builder.gold_builder import GoldBuilder
from structverify.eval.builder.text_utils import (
    build_claim_text_from_spec,
    claim_text_reflects_gold_value,
    normalize_kosis_unit,
    prose_has_markdown_emphasis,
    strip_llm_markdown_emphasis,
)
from structverify.eval.builder.unverifiable_recipes import (
    DETECTION_FRIENDLY_RECIPES,
    pick_unverifiable_recipe,
)
from structverify.eval.harness.runner import merge_harness_config
from structverify.eval.builder.spec_preflight import (
    validate_specs_before_prose,
)
from structverify.eval.builder.validator import EvalArticleValidator
from structverify.eval.builder.dataset_writer import DatasetWriter


def test_all_registry_domains_have_keywords():
    domains = all_eval_domains(list(DOMAIN_CATEGORY_KEYWORDS.keys()))
    assert len(domains) == 12
    for d in domains:
        assert get_category_keywords(d), f"missing keywords for {d}"


def test_healthcare_keyword_not_health():
    assert "healthcare" in DOMAIN_CATEGORY_KEYWORDS
    assert "health" not in DOMAIN_CATEGORY_KEYWORDS


def test_perturb_value_differs():
    rng = random.Random(42)
    official = 100.0
    perturbed = _perturb_value(official, rng)
    assert perturbed != official


def test_perturb_time_period():
    rng = random.Random(1)
    out = _perturb_time_period("2023", rng)
    assert out in ("2022", "2024")


def test_allocate_verdict_targets():
    q = allocate_verdict_targets(360, {"match": 0.45, "mismatch": 0.35, "unverifiable": 0.20})
    assert sum(q.values()) == 360
    # rounding remainder → largest bucket may be ±1
    assert 161 <= q["match"] <= 163
    assert 125 <= q["mismatch"] <= 127
    assert 71 <= q["unverifiable"] <= 73


def test_allocate_domain_quotas():
    domains = ["economy", "population", "finance"]
    density = {"economy": 100, "population": 50, "finance": 50}
    q = KosisRowSampler.allocate_domain_quotas(domains, 12, density)
    assert sum(q.values()) == 12
    assert q["economy"] >= q["population"]


def test_allocate_domain_quotas_from_shares_pilot():
    domains = [
        "economy",
        "population",
        "healthcare",
        "policy",
        "employment",
        "education",
        "environment",
        "agriculture",
        "real_estate",
        "finance",
        "automotive_technology",
        "weather",
    ]
    shares = {
        "economy": 0.125,
        "population": 0.05,
        "healthcare": 0.125,
        "policy": 0.05,
        "employment": 0.075,
        "education": 0.075,
        "environment": 0.075,
        "agriculture": 0.075,
        "real_estate": 0.075,
        "finance": 0.10,
        "automotive_technology": 0.075,
        "weather": 0.10,
    }
    q = KosisRowSampler.allocate_domain_quotas(
        domains, 40, {}, domain_shares=shares
    )
    assert sum(q.values()) == 40
    assert q["population"] == 2
    assert q["policy"] == 2
    assert q["economy"] == 5


def test_allocate_domain_quotas_from_shares_v1_120():
    shares = {"economy": 0.5, "finance": 0.5}
    q = KosisRowSampler.allocate_domain_quotas(
        ["economy", "finance"], 120, {}, domain_shares=shares
    )
    assert sum(q.values()) == 120
    assert q["economy"] == 60
    assert q["finance"] == 60


def test_validator_rejects_mismatch_equal_official_value_recipe():
    state = BuildState(dataset_id="t", mode="pilot", seed=1)
    article = EvalArticle(
        article_id="a1",
        intended_domain="economy",
        article_text="x" * 100,
        claims=[
            EvalClaim(
                claim_id="c1",
                claim_text="2023 GDP는 100% 증가했다",
                gold_schema=GoldSchema(
                    indicator="GDP",
                    value=100.0,
                    unit="%",
                    time_period="2023",
                ),
                gold_stat_id="DT_TEST",
                gold_official_value=100.0,
                gold_verdict="mismatch",
                mismatch_recipe="value",
            )
        ],
    )
    ok, errors = EvalArticleValidator().validate(
        article,
        state,
        domain_quota={"economy": 10},
        verdict_quota={"match": 10, "mismatch": 10, "unverifiable": 10},
    )
    assert not ok
    assert any("mismatch value equals official" in e for e in errors)


def test_build_state_failed_stat_dedup():
    state = BuildState(dataset_id="t", mode="pilot", seed=1)
    state.register_failed_stat("DT_BAD")
    state.register_failed_stat("DT_BAD")
    assert state.failed_stat_id_set() == {"DT_BAD"}


def test_latest_year_from_stat_rec():
    from structverify.retrieval.base_connector import StatRecord

    rec = StatRecord(
        stat_id="DT_X",
        stat_name="test",
        org_name="org",
        metadata={
            "getMeta_PRD": {"row": [{"PRD_DE": "2018"}, {"PRD_DE": "2019"}]}
        },
    )
    assert _latest_year_from_stat_rec(rec) == "2019"


def test_validator_rejects_duplicate_claim_text():
    state = BuildState(dataset_id="t", mode="pilot", seed=1)
    text = "# 제목\n\n날짜: 2024-01-01\n\n" + ("동일 문장. " * 20)
    article = EvalArticle(
        article_id="a1",
        intended_domain="economy",
        article_text=text + "2023년 GDP는 2.6%였다.",
        claims=[
            EvalClaim(
                claim_id="c1",
                claim_text="2023년 GDP는 2.6%였다.",
                gold_schema=GoldSchema(
                    indicator="GDP", value=2.6, unit="%", time_period="2023"
                ),
                gold_stat_id="DT_X",
                gold_official_value=2.6,
                gold_verdict="match",
            ),
            EvalClaim(
                claim_id="c2",
                claim_text="2023년 GDP는 2.6%였다.",
                gold_schema=GoldSchema(
                    indicator="GDP", value=3.0, unit="%", time_period="2023"
                ),
                gold_stat_id="DT_X",
                gold_official_value=2.6,
                gold_verdict="mismatch",
            ),
        ],
    )
    ok, errors = EvalArticleValidator().validate(
        article,
        state,
        domain_quota={"economy": 10},
        verdict_quota={"match": 10, "mismatch": 10, "unverifiable": 10},
    )
    assert not ok
    assert any("duplicate claim_text" in e for e in errors)


def test_strip_llm_markdown_emphasis_removes_bold():
    raw = "비율은 **11.0%** 를 차지하며, 면적은 **692.81km²**이다."
    cleaned = strip_llm_markdown_emphasis(raw)
    assert "**" not in cleaned
    assert "11.0%" in cleaned
    assert "692.81km²" in cleaned
    assert not prose_has_markdown_emphasis(cleaned)


def test_validator_rejects_markdown_bold_in_claim():
    state = BuildState(dataset_id="t", mode="pilot", seed=1)
    body = "# 제목\n\n" + ("본문. " * 30) + "2023년 지표는 **2.6%**였다."
    article = EvalArticle(
        article_id="a1",
        intended_domain="economy",
        article_text=body,
        claims=[
            EvalClaim(
                claim_id="c1",
                claim_text="2023년 지표는 **2.6%**였다.",
                gold_schema=GoldSchema(
                    indicator="GDP", value=2.6, unit="%", time_period="2023"
                ),
                gold_stat_id="DT_X",
                gold_official_value=2.6,
                gold_verdict="match",
            )
        ],
    )
    ok, errors = EvalArticleValidator().validate(
        article,
        state,
        domain_quota={"economy": 10},
        verdict_quota={"match": 10, "mismatch": 10, "unverifiable": 10},
    )
    assert not ok
    assert any("markdown emphasis" in e for e in errors)


def test_normalize_kosis_unit_strips_percent_suffix():
    assert normalize_kosis_unit("백만원%") == "백만원"
    assert normalize_kosis_unit("개%") == "개"


def test_claim_text_reflects_gold_value():
    assert claim_text_reflects_gold_value("2023년 지표는 2.6%였다.", 2.6, "%")
    assert not claim_text_reflects_gold_value("2023년 지표는 3.0%였다.", 2.6, "%")


def test_story_anchor_region_extraction():
    path = "MT_ZTITLE>지자체>경기도>안성시>시사회>인구"
    assert extract_region_token(path) == "안성시"


def test_extract_region_skips_household_segment():
    path = "MT_ZTITLE > 인구 > 가구 > 1인 가구 비율"
    assert extract_region_token(path) is None


def test_extract_region_skips_family_household_phrase():
    path = "MT_ZTITLE > 인구 > 가족과가구 > 지표"
    assert extract_region_token(path) is None


def test_normalize_region_token_strips_province_prefix():
    assert normalize_region_token("경기도이천시") == "이천시"
    assert normalize_region_token("경상남도사천시") == "사천시"
    assert normalize_region_token("전북특별자치도진안군") == "진안군"
    assert normalize_region_token("광주광역시") == "광주광역시"
    assert normalize_region_token("청도군") == "청도군"


def test_extract_region_concatenated_segment():
    path = "MT_ZTITLE>지자체>경기도이천시>시사회>인구"
    assert extract_region_token(path) == "이천시"


def test_extract_region_prefers_city_over_province_segment():
    path = "MT_ZTITLE>지자체>경기도>이천시>시사회>인구"
    assert extract_region_token(path) == "이천시"


def test_specs_single_region_rejects_second_city():
    national = ClaimSpec(
        claim_id="c0",
        intended_verdict="match",
        gold_schema=GoldSchema(
            indicator="GDP", value=1.0, unit="%", time_period="2023"
        ),
        gold_stat_id="DT_N",
        gold_evidence=GoldEvidence(category_path="MT_ZTITLE > 경제 > 국민계정"),
    )
    local_a = ClaimSpec(
        claim_id="c1",
        intended_verdict="match",
        gold_schema=GoldSchema(
            indicator="a", value=1.0, unit="%", time_period="2021"
        ),
        gold_stat_id="DT_A",
        gold_evidence=GoldEvidence(
            category_path="MT>지자체>경기도>이천시>교육"
        ),
    )
    local_b = ClaimSpec(
        claim_id="c2",
        intended_verdict="match",
        gold_schema=GoldSchema(
            indicator="b", value=2.0, unit="%", time_period="2021"
        ),
        gold_stat_id="DT_B",
        gold_evidence=GoldEvidence(
            category_path="MT>지자체>경상남도>사천시>교육"
        ),
    )
    assert specs_single_region([national])
    assert specs_single_region([national], local_a)
    assert not specs_single_region([national, local_a], local_b)
    assert region_token_for_spec(local_a) == "이천시"


def test_is_story_lock_region_excludes_province_only():
    assert is_story_lock_region("이천시")
    assert is_story_lock_region("광주광역시")
    assert not is_story_lock_region("강원특별자치도")
    assert not is_story_lock_region("경기도")


def test_pick_viable_region_pool_requires_min_rows():
    rows = [
        {"stat_id": "A", "category_path": "MT>지자체>경기도>이천시>인구"},
        {"stat_id": "B", "category_path": "MT>지자체>경기도>이천시>교육"},
        {"stat_id": "C", "category_path": "MT>지자체>경상남도>사천시>인구"},
    ]
    rng = random.Random(0)
    picked = pick_viable_region_pool(rows, 2, rng)
    assert picked is not None
    region, pool = picked
    assert region == "이천시"
    assert len(pool) == 2


def test_catalog_path_is_national_vs_local():
    assert catalog_path_is_national("MT_ZTITLE > 증권·파생상품시장통계 > 파생상품")
    assert not catalog_path_is_national(
        "MT_ZTITLE > 지자체 기본통계 > 경기도 > 경기도포천시기본통계 > 주택"
    )
    assert not catalog_path_is_national(
        "MT_ZTITLE > 인구 및 사회(사회조사) > 대구광역시 > 대구광역시사회조사"
    )


def test_pick_viable_national_pool():
    rows = [
        {"stat_id": "N1", "category_path": "MT_ZTITLE > 증권·파생상품시장통계 > A"},
        {"stat_id": "N2", "category_path": "MT_ZTITLE > 한국교육고용패널조사 > B"},
        {"stat_id": "L1", "category_path": "MT>지자체>경기도>이천시>인구"},
    ]
    pool = pick_viable_national_pool(rows, 2, random.Random(0))
    assert pool is not None
    assert len(pool) == 2
    assert all(catalog_row_is_national(r) for r in pool)


def test_pick_domain_fallback_excludes_tried_and_uses_soft_quota():
    builder = EvalSetBuilder.__new__(EvalSetBuilder)
    builder.rng = random.Random(0)
    builder.quota_tolerance = 2
    builder.domain_fallback_soft_quota = True
    domain_quota = {"economy": 5, "healthcare": 5, "finance": 4}
    state = BuildState(
        dataset_id="t",
        mode="pilot",
        seed=1,
        domain_counts={"economy": 5, "healthcare": 4, "finance": 4},
    )
    tried = {"economy", "healthcare"}
    domain = builder._pick_domain_fallback(domain_quota, state, tried)
    assert domain == "finance"


def test_domain_candidates_soft_quota_allows_slight_overflow():
    builder = EvalSetBuilder.__new__(EvalSetBuilder)
    builder.quota_tolerance = 2
    domain_quota = {"finance": 4}
    state = BuildState(
        dataset_id="t",
        mode="pilot",
        seed=1,
        domain_counts={"finance": 4},
    )
    strict = builder._domain_candidates(domain_quota, state, soft_quota=False)
    soft = builder._domain_candidates(domain_quota, state, soft_quota=True)
    assert strict == []
    assert soft == ["finance"]


def test_bootstrap_national_first_prefers_national_pool():
    builder = EvalSetBuilder.__new__(EvalSetBuilder)
    builder.claims_min = 2
    builder.scope_strategy = "national_first"
    builder.single_region_per_article = True
    builder.rng = random.Random(0)
    rows = [
        {"stat_id": "N1", "category_path": "MT_ZTITLE > 증권·파생상품시장통계 > A"},
        {"stat_id": "N2", "category_path": "MT_ZTITLE > 한국교육고용패널조사 > B"},
        {"stat_id": "L1", "category_path": "MT>지자체>경기도>이천시>인구"},
        {"stat_id": "L2", "category_path": "MT>지자체>경기도>이천시>주택"},
    ]
    boot = builder._bootstrap_article_scope(rows, "eval_test_0001")
    assert boot is not None
    scope, pool, _anchor, locked = boot
    assert scope == "national"
    assert len(pool) == 2
    assert locked == set()


def test_bootstrap_national_first_falls_back_to_local():
    builder = EvalSetBuilder.__new__(EvalSetBuilder)
    builder.claims_min = 2
    builder.scope_strategy = "national_first"
    builder.single_region_per_article = True
    builder.rng = random.Random(0)
    rows = [
        {"stat_id": "L1", "category_path": "MT>지자체>경기도>이천시>인구"},
        {"stat_id": "L2", "category_path": "MT>지자체>경기도>이천시>주택"},
    ]
    boot = builder._bootstrap_article_scope(rows, "eval_test_0002")
    assert boot is not None
    scope, pool, _anchor, locked = boot
    assert scope == "local"
    assert len(pool) == 2
    assert locked == {"이천시"}


def test_spec_preflight_national_rejects_local_region_tokens():
    specs = [
        ClaimSpec(
            claim_id="c1",
            intended_verdict="match",
            gold_schema=GoldSchema(
                indicator="a", value=1.0, unit="%", time_period="2021"
            ),
            gold_stat_id="DT_A",
            gold_evidence=GoldEvidence(
                category_path="MT>지자체>경기도>이천시>인구"
            ),
        ),
    ]
    ok, errors = validate_specs_before_prose(
        specs, article_scope="national"
    )
    assert not ok
    assert any("national article" in e for e in errors)


def test_locked_regions_blocks_other_city_catalog_row():
    anchor = StoryAnchor(region_token="이천시", is_local=True, anchor_year=2021)
    locked = {"이천시"}
    ok_row = {"category_path": "MT>지자체>경기도>이천시>인구", "stat_id": "A"}
    bad_row = {"category_path": "MT>지자체>경상남도>사천시>인구", "stat_id": "B"}
    assert catalog_row_compatible(ok_row, anchor, locked_regions=locked)
    assert not catalog_row_compatible(bad_row, anchor, locked_regions=locked)


def test_spec_preflight_allows_same_city_concat_vs_split_path():
    base = "MT_ZTITLE > 인구 및 사회(사회조사) > 경기도"
    specs = [
        ClaimSpec(
            claim_id="c1",
            intended_verdict="match",
            gold_schema=GoldSchema(
                indicator="a", value=1.0, unit="%", time_period="2021"
            ),
            gold_stat_id="DT_A",
            gold_evidence=GoldEvidence(category_path=f"{base} > 이천시 > 교육"),
        ),
        ClaimSpec(
            claim_id="c2",
            intended_verdict="match",
            gold_schema=GoldSchema(
                indicator="b", value=2.0, unit="%", time_period="2021"
            ),
            gold_stat_id="DT_B",
            gold_evidence=GoldEvidence(
                category_path="MT_ZTITLE > 인구 및 사회(사회조사) > 경기도이천시 > 보건"
            ),
        ),
    ]
    ok, errors = validate_specs_before_prose(specs)
    assert ok, errors


def test_validator_allows_mismatch_time_same_value():
    state = BuildState(dataset_id="t", mode="pilot", seed=1)
    body = "# 제목\n\n" + ("본문. " * 30) + "2022년 GDP는 2.6%였다."
    article = EvalArticle(
        article_id="a1",
        intended_domain="economy",
        article_text=body,
        claims=[
            EvalClaim(
                claim_id="c1",
                claim_text="2022년 GDP는 2.6%였다.",
                gold_schema=GoldSchema(
                    indicator="GDP", value=2.6, unit="%", time_period="2022"
                ),
                gold_stat_id="DT_X",
                gold_official_value=2.6,
                gold_verdict="mismatch",
                mismatch_recipe="time",
            )
        ],
    )
    ok, errors = EvalArticleValidator().validate(
        article,
        state,
        domain_quota={"economy": 10},
        verdict_quota={"match": 10, "mismatch": 10, "unverifiable": 10},
    )
    assert ok, errors


def test_spec_preflight_rejects_cross_region_specs():
    specs = [
        ClaimSpec(
            claim_id="c1",
            intended_verdict="match",
            gold_schema=GoldSchema(
                indicator="a", value=1.5, unit="%", time_period="2021"
            ),
            gold_stat_id="DT_A",
            gold_evidence=GoldEvidence(
                category_path="MT_ZTITLE > 인구 및 사회(사회조사) > 경상남도 > 창녕군 > 교육"
            ),
        ),
        ClaimSpec(
            claim_id="c2",
            intended_verdict="match",
            gold_schema=GoldSchema(
                indicator="b", value=2.1, unit="%", time_period="2021"
            ),
            gold_stat_id="DT_B",
            gold_evidence=GoldEvidence(
                category_path="MT_ZTITLE > 인구 및 사회(사회조사) > 강원 > 철원군 > 교육"
            ),
        ),
    ]
    ok, errors = validate_specs_before_prose(specs)
    assert not ok
    assert any("cross-region" in e for e in errors)


def test_spec_compatible_rejects_distant_year():
    anchor = StoryAnchor(anchor_year=2023, is_local=False)
    spec = ClaimSpec(
        claim_id="c1",
        intended_verdict="match",
        gold_schema=GoldSchema(
            indicator="x", value=1.0, unit="%", time_period="2020"
        ),
        gold_stat_id="DT_A",
        gold_evidence=GoldEvidence(category_path="MT>경제>지표"),
    )
    assert not spec_compatible_with_anchor(spec, anchor, year_slack=1)
    spec_ok = ClaimSpec(
        claim_id="c2",
        intended_verdict="match",
        gold_schema=GoldSchema(
            indicator="x", value=1.0, unit="%", time_period="2023"
        ),
        gold_stat_id="DT_A",
        gold_evidence=GoldEvidence(category_path="MT>경제>지표"),
    )
    assert spec_compatible_with_anchor(spec_ok, anchor, year_slack=1)


def test_path_is_local_social_survey_path():
    path = "MT_ZTITLE > 인구 및 사회(사회조사) > 경상남도 > 창녕군 > 교육"
    assert path_is_local(path)


def test_region_compatible_rejects_cross_county():
    anchor = StoryAnchor(region_token="창녕군", anchor_year=2021, is_local=True)
    ok_path = "MT_ZTITLE > 인구 및 사회(사회조사) > 경상남도 > 창녕군 > 교육"
    bad_path = "MT_ZTITLE > 인구 및 사회(사회조사) > 강원특별자치도 > 철원군 > 교육"
    assert region_compatible_with_anchor(anchor, ok_path)
    assert not region_compatible_with_anchor(anchor, bad_path)


def test_spec_compatible_rejects_cross_region_social_survey():
    anchor = StoryAnchor.from_spec(
        ClaimSpec(
            claim_id="c0",
            intended_verdict="match",
            gold_schema=GoldSchema(
                indicator="a", value=1.0, unit="%", time_period="2021"
            ),
            gold_stat_id="DT_A",
            gold_evidence=GoldEvidence(
                category_path="MT_ZTITLE > 인구 및 사회(사회조사) > 경상남도 > 창녕군 > 교육"
            ),
        )
    )
    other = ClaimSpec(
        claim_id="c1",
        intended_verdict="match",
        gold_schema=GoldSchema(
            indicator="b", value=2.0, unit="%", time_period="2021"
        ),
        gold_stat_id="DT_B",
        gold_evidence=GoldEvidence(
            category_path="MT_ZTITLE > 인구 및 사회(사회조사) > 강원특별자치도 > 철원군 > 교육"
        ),
    )
    assert not spec_compatible_with_anchor(other, anchor, single_region=True)


def test_gold_builder_probe_cache_hit():
    gb = GoldBuilder(config={"kosis": {"probe_cache_enabled": True}}, seed=1)
    row = {"stat_id": "DT_TEST", "stat_name": "t", "org_id": "113", "category_path": "a>b"}
    gb._probe_cache["DT_TEST"] = {
        "stat_id": "DT_TEST",
        "stat_name": "t",
        "org_name": "",
        "category_path": "a>b",
        "official_value": 1.0,
        "unit": "%",
        "time_period": "2023",
        "indicator": "t",
    }
    import asyncio

    probed = asyncio.run(gb.fetch_probe(row))
    assert probed is not None
    assert probed["official_value"] == 1.0


def test_validator_rejects_cross_region_article():
    state = BuildState(dataset_id="t", mode="pilot", seed=1)
    body = "# 제목\n\n" + ("본문. " * 30)
    body += "2021년 창녕군 지표는 1.5%였다. 2021년 철원군 지표는 2.1%였다."
    article = EvalArticle(
        article_id="a1",
        intended_domain="education",
        article_text=body,
        claims=[
            EvalClaim(
                claim_id="c1",
                claim_text="2021년 창녕군 지표는 1.5%였다.",
                gold_schema=GoldSchema(
                    indicator="a", value=1.5, unit="%", time_period="2021"
                ),
                gold_stat_id="DT_A",
                gold_official_value=1.5,
                gold_verdict="match",
                gold_evidence=GoldEvidence(
                    category_path="MT_ZTITLE > 인구 및 사회(사회조사) > 경상남도 > 창녕군 > 교육"
                ),
            ),
            EvalClaim(
                claim_id="c2",
                claim_text="2021년 철원군 지표는 2.1%였다.",
                gold_schema=GoldSchema(
                    indicator="b", value=2.1, unit="%", time_period="2021"
                ),
                gold_stat_id="DT_B",
                gold_official_value=2.1,
                gold_verdict="match",
                gold_evidence=GoldEvidence(
                    category_path="MT_ZTITLE > 인구 및 사회(사회조사) > 강원 > 철원군 > 교육"
                ),
            ),
        ],
    )
    ok, errors = EvalArticleValidator(reject_cross_region=True).validate(
        article,
        state,
        domain_quota={"education": 10},
        verdict_quota={"match": 10, "mismatch": 10, "unverifiable": 10},
    )
    assert not ok
    assert any("cross-region" in e for e in errors)


def test_catalog_row_compatible_with_local_anchor():
    anchor = StoryAnchor(region_token="안성시", is_local=True, anchor_year=2023)
    ok_row = {"category_path": "MT>지자체>경기도>안성시>시사회>인구", "stat_id": "A"}
    bad_row = {"category_path": "MT>지자체>대구광역시>달서구>시사회>인구", "stat_id": "B"}
    assert catalog_row_compatible(ok_row, anchor)
    assert not catalog_row_compatible(bad_row, anchor)


def test_validator_rejects_claim_text_missing_gold_value():
    state = BuildState(dataset_id="t", mode="pilot", seed=1)
    body = "# 제목\n\n" + ("본문. " * 30) + "2023년 GDP는 99%였다."
    article = EvalArticle(
        article_id="a1",
        intended_domain="economy",
        article_text=body,
        claims=[
            EvalClaim(
                claim_id="c1",
                claim_text="2023년 GDP는 99%였다.",
                gold_schema=GoldSchema(
                    indicator="GDP", value=2.6, unit="%", time_period="2023"
                ),
                gold_stat_id="DT_X",
                gold_official_value=2.6,
                gold_verdict="match",
            )
        ],
    )
    ok, errors = EvalArticleValidator().validate(
        article,
        state,
        domain_quota={"economy": 10},
        verdict_quota={"match": 10, "mismatch": 10, "unverifiable": 10},
    )
    assert not ok
    assert any("missing gold_schema.value" in e for e in errors)


def test_dataset_writer_roundtrip(tmp_path: Path):
    writer = DatasetWriter(tmp_path, "test_set")
    article = EvalArticle(
        article_id="eval_economy_0001",
        intended_domain="economy",
        article_text="테스트 기사 본문입니다. " * 10,
        claims=[
            EvalClaim(
                claim_id="eval_economy_0001_c01",
                claim_text="2023년 경제성장률은 2.6%였다.",
                gold_schema=GoldSchema(
                    indicator="경제성장률",
                    value=2.6,
                    unit="%",
                    time_period="2023",
                ),
                gold_stat_id="DT_X",
                gold_official_value=2.6,
                gold_verdict="match",
            )
        ],
    )
    writer.append_article(article)
    loaded = writer.load_articles()
    assert len(loaded) == 1
    assert loaded[0].article_id == article.article_id
    assert loaded[0].claims[0].gold_verdict == "match"

    manifest = EvalManifest(
        dataset_id="test_set",
        mode="pilot",
        article_count=1,
        claim_count=1,
    )
    writer.write_manifest(manifest, {"mode": "pilot"}, status="frozen")
    data = json.loads((tmp_path / "test_set" / "manifest.json").read_text())
    assert data["status"] == "frozen"
    assert data["articles_sha256"]


def test_build_claim_text_news_varied_rotates_by_claim_id():
    sch = GoldSchema(
        indicator="출생아 수",
        value=100.0,
        unit="명",
        time_period="2024",
        population="전국",
    )
    texts = {
        build_claim_text_from_spec(
            ClaimSpec(claim_id=f"c{i}", intended_verdict="match", gold_schema=sch),
            style="news_varied",
        )
        for i in range(6)
    }
    assert len(texts) >= 2
    assert not all("집계됐다" in t for t in texts)


def test_build_claim_text_news_style():
    spec = ClaimSpec(
        claim_id="c1",
        intended_verdict="match",
        gold_schema=GoldSchema(
            indicator="4월 출생아 수",
            value=21717.0,
            unit="명",
            time_period="2025",
            population="전국",
        ),
    )
    news = build_claim_text_from_spec(spec, style="news")
    caption = build_claim_text_from_spec(spec, style="caption")
    assert "나타났다" not in news
    assert "집계됐다" in news
    assert "나타났다" in caption
    assert claim_text_reflects_gold_value(news, 21717.0, "명")


def test_pick_unverifiable_recipe_allowed_subset():
    rng = random.Random(0)
    for _ in range(20):
        r = pick_unverifiable_recipe(rng, allowed=list(DETECTION_FRIENDLY_RECIPES))
        assert r in ("U2", "U3")


def test_merge_harness_config_candidate_detection():
    merged = merge_harness_config(
        {"candidate_detection": {"threshold": 0.5}, "persist_to_db": True}
    )
    assert merged["candidate_detection"]["threshold"] == 0.5
    assert merged["persist_to_db"] is True


def test_ensure_headline_blank_line():
    from structverify.eval.builder.article_template import ensure_headline_blank_line

    bad = "# 제목\n본문 시작"
    fixed = ensure_headline_blank_line(bad)
    assert fixed.startswith("# 제목\n\n본문")


def test_assemble_template_article_structure():
    from structverify.eval.builder.article_template import assemble_template_article

    spec = ClaimSpec(
        claim_id="c1",
        intended_verdict="match",
        gold_schema=GoldSchema(
            indicator="출생아 수",
            value=100.0,
            unit="명",
            time_period="2024",
            population="전국",
        ),
        gold_evidence=GoldEvidence(
            category_path="MT_ZTITLE > 인구통계 > 출생",
        ),
    )
    claim = build_claim_text_from_spec(spec, style="news")
    body = assemble_template_article("economy", [spec], [claim])
    assert body.startswith("# ")
    assert "연도=" not in body.split("\n")[0]
    assert "\n\n\n\n" not in body
    assert claim in body
    assert "100" not in body.split(claim)[0]


def test_human_headline_not_metadata():
    from structverify.eval.builder.article_template import human_headline_from_specs

    spec = ClaimSpec(
        claim_id="c1",
        intended_verdict="match",
        gold_schema=GoldSchema(
            indicator="미충족 의료율",
            value=1.0,
            unit="명",
            time_period="2022",
        ),
        gold_evidence=GoldEvidence(
            category_path="MT_ZTITLE > 한국의료패널조사 > 2019년 이후",
        ),
    )
    title = human_headline_from_specs("healthcare", [spec])
    assert "연도=" not in title
    assert "2022" in title
    assert "한국의료패널조사" in title


def test_validator_rejects_banned_claim_phrasing():
    from structverify.eval.builder.validator import EvalArticleValidator
    from structverify.eval.builder.schemas import EvalClaim, EvalArticle, GoldSchema

    v = EvalArticleValidator(reject_banned_claim_phrasing=True)
    article = EvalArticle(
        article_id="a1",
        intended_domain="agriculture",
        article_scope="national",
        article_text="# t\n\n2022년 전국 농어업인은 연간 4,000회에 걸쳐 이용할 수 있다.",
        claims=[
            EvalClaim(
                claim_id="c1",
                claim_text="2022년 전국 농어업인은 연간 4,000회에 걸쳐 이용할 수 있다.",
                gold_verdict="match",
                gold_schema=GoldSchema(
                    indicator="x",
                    value=4000.0,
                    unit="회",
                    time_period="2022",
                ),
            )
        ],
    )
    ok, errors = v.validate(article, BuildState(), {}, {}, quota_tolerance=99)
    assert not ok
    assert any("banned phrasing" in e for e in errors)


def test_validator_rejects_lead_gold_values():
    from structverify.eval.builder.validator import EvalArticleValidator
    from structverify.eval.builder.schemas import EvalClaim, EvalArticle, GoldSchema

    claim = "2024년 전국 출생아 수는 2,171명으로 집계됐다."
    article = EvalArticle(
        article_id="a1",
        intended_domain="population",
        article_scope="national",
        article_text=(
            f"# 제목\n\n리드에 2,171명이 언급됐다.\n\n{claim}\n\n마무리."
        ),
        claims=[
            EvalClaim(
                claim_id="c1",
                claim_text=claim,
                gold_verdict="match",
                gold_schema=GoldSchema(
                    indicator="출생아",
                    value=2171.0,
                    unit="명",
                    time_period="2024",
                    population="전국",
                ),
            )
        ],
    )
    v = EvalArticleValidator(
        require_headline_blank_line=True,
        reject_lead_gold_values=True,
    )
    ok, errors = v.validate(article, BuildState(), {}, {}, quota_tolerance=99)
    assert not ok
    assert any("lead paragraphs" in e for e in errors)


def test_force_deterministic_claim_text_overwrites_llm():
    from structverify.eval.builder.prose_filler import LLMProseFiller

    spec = ClaimSpec(
        claim_id="c1",
        intended_verdict="match",
        gold_schema=GoldSchema(
            indicator="출생아 수",
            value=2171.0,
            unit="명",
            time_period="2024",
            population="전국",
        ),
    )
    filler = LLMProseFiller(
        config={
            "prose": {
                "claim_text_style": "news",
                "force_deterministic_claim_text": True,
            }
        }
    )
    out = filler._finalize_claim_texts(
        [spec],
        {"c1": "2024년 출생아는 999명으로 나타났다."},
    )
    assert "집계됐다" in out["c1"]
    assert "999" not in out["c1"]
    assert claim_text_reflects_gold_value(out["c1"], 2171.0, "명")


@pytest.mark.asyncio
async def test_detection_preflight_mocked():
    from unittest.mock import AsyncMock, patch

    from structverify.core.schemas import Claim
    from structverify.eval.builder.detection_preflight import validate_article_detection
    from uuid import uuid4

    gold = "2024년 전국 출생아 수는 2,171명으로 집계됐다."
    body = f"# 제목\n\n리드.\n\n{gold}\n\n끝."
    mock_claim = Claim(
        doc_id=uuid4(),
        block_id="b0000",
        sent_id="b0000_s0000",
        claim_text=gold,
    )

    with patch(
        "structverify.eval.builder.detection_preflight.classify_domain",
        new=AsyncMock(return_value=("population", "desc")),
    ), patch(
        "structverify.eval.builder.detection_preflight.detect_claims",
        new=AsyncMock(return_value=[mock_claim]),
    ):
        ok, errors = await validate_article_detection(
            body, [gold], {}, min_claims=1, min_gold_matches=1
        )
    assert ok
    assert errors == []
