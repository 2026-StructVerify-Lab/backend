"""Unverifiable claim recipes (U1–U5) for eval quota."""
from __future__ import annotations

import random
from typing import Literal

from structverify.eval.builder.schemas import ClaimSpec, GoldSchema, UnverifiableRecipe

RECIPE_WEIGHTS: dict[UnverifiableRecipe, float] = {
    "U1": 0.08,
    "U2": 0.05,
    "U3": 0.04,
    "U4": 0.02,
    "U5": 0.01,
}

RECIPE_REASONS: dict[UnverifiableRecipe, str] = {
    "U1": "forecast_or_ranking_not_in_kosis",
    "U2": "domain_relevant_but_no_catalog_match",
    "U3": "kosis_fetch_unavailable",
    "U4": "ambiguous_multi_topic",
    "U5": "vague_unit_or_time",
}


# E2E 탐지 친화 eval: U1(전망/순위), U4(복합), U5(모호 시점) 제외
DETECTION_FRIENDLY_RECIPES: tuple[UnverifiableRecipe, ...] = ("U2", "U3")


def pick_unverifiable_recipe(
    rng: random.Random,
    allowed: list[UnverifiableRecipe] | None = None,
) -> UnverifiableRecipe:
    if allowed is not None:
        recipes = [r for r in allowed if r in RECIPE_WEIGHTS]
        if not recipes:
            raise ValueError("pick_unverifiable_recipe: allowed recipes empty")
        weights = [RECIPE_WEIGHTS[r] for r in recipes]
        return rng.choices(recipes, weights=weights, k=1)[0]
    recipes = list(RECIPE_WEIGHTS.keys())
    weights = [RECIPE_WEIGHTS[r] for r in recipes]
    return rng.choices(recipes, weights=weights, k=1)[0]


def build_unverifiable_spec(
    claim_id: str,
    domain: str,
    recipe: UnverifiableRecipe,
    rng: random.Random,
    catalog_row: dict | None = None,
) -> ClaimSpec:
    reason = RECIPE_REASONS[recipe]

    if recipe == "U1":
        schema = GoldSchema(
            indicator="시장 전망",
            value=None,
            unit=None,
            time_period=str(rng.randint(2025, 2027)),
            population="전체",
        )
        return ClaimSpec(
            claim_id=claim_id,
            intended_verdict="unverifiable",
            gold_schema=schema,
            unverifiable_reason=reason,
            unverifiable_recipe=recipe,
        )

    if recipe == "U2":
        schema = GoldSchema(
            indicator=f"{domain} 비공식 지표",
            value=float(rng.randint(10, 99)),
            unit="%",
            time_period="2024",
            population="전체",
        )
        return ClaimSpec(
            claim_id=claim_id,
            intended_verdict="unverifiable",
            gold_schema=schema,
            unverifiable_reason=reason,
            unverifiable_recipe=recipe,
        )

    if recipe == "U3" and catalog_row:
        return ClaimSpec(
            claim_id=claim_id,
            intended_verdict="unverifiable",
            gold_schema=GoldSchema(
                indicator=catalog_row.get("stat_name", "unknown"),
                value=None,
                unit=None,
                time_period="2024",
            ),
            gold_stat_id=catalog_row.get("stat_id"),
            unverifiable_reason=reason,
            unverifiable_recipe=recipe,
            catalog_row=catalog_row,
        )

    if recipe == "U4":
        return ClaimSpec(
            claim_id=claim_id,
            intended_verdict="unverifiable",
            gold_schema=GoldSchema(
                indicator="복합 정책 효과",
                value=float(rng.randint(1, 20)),
                unit="%p",
                time_period="2024",
            ),
            unverifiable_reason=reason,
            unverifiable_recipe=recipe,
        )

    return ClaimSpec(
        claim_id=claim_id,
        intended_verdict="unverifiable",
        gold_schema=GoldSchema(
            indicator="규모",
            value=float(rng.randint(100, 999)),
            unit="정도",
            time_period="최근",
        ),
        unverifiable_reason=RECIPE_REASONS["U5"],
        unverifiable_recipe="U5",
    )


def allocate_verdict_targets(
    total_claims: int,
    verdict_ratios: dict[str, float],
) -> dict[str, int]:
    keys = ["match", "mismatch", "unverifiable"]
    raw = {k: total_claims * float(verdict_ratios.get(k, 0)) for k in keys}
    quotas = {k: int(raw[k]) for k in keys}
    remainder = total_claims - sum(quotas.values())
    order: list[Literal["match", "mismatch", "unverifiable"]] = [
        "match",
        "mismatch",
        "unverifiable",
    ]
    i = 0
    while remainder > 0:
        quotas[order[i % len(order)]] += 1
        remainder -= 1
        i += 1
    return quotas
