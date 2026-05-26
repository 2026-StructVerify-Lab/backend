"""Validate claim specs before LLM prose (fail fast, save tokens/API)."""
from __future__ import annotations

from structverify.eval.builder.schemas import ClaimSpec
from structverify.eval.builder.story_coherence import story_region_tokens_from_specs


def validate_specs_before_prose(
    specs: list[ClaimSpec],
    *,
    reject_cross_region: bool = True,
    article_scope: str = "local",
) -> tuple[bool, list[str]]:
    """
    LLM/KOSIS prose 전 스펙 조합 검사.

    article validator와 동일한 cross-region 규칙을 specs에 적용한다.
    """
    errors: list[str] = []

    if not specs:
        errors.append("no specs")
        return False, errors

    seen_match_fact: set[tuple[str, str]] = set()
    for spec in specs:
        if spec.intended_verdict in ("match", "mismatch"):
            sch = spec.gold_schema
            if not sch or sch.value is None:
                errors.append(f"{spec.claim_id}: verifiable spec missing value")
            if sch and not (sch.time_period or "").strip():
                errors.append(f"{spec.claim_id}: verifiable spec missing time_period")
            if (
                spec.intended_verdict == "mismatch"
                and spec.mismatch_recipe == "value"
                and sch
                and spec.gold_official_value is not None
                and sch.value is not None
                and abs(float(sch.value) - float(spec.gold_official_value)) < 1e-9
            ):
                errors.append(
                    f"{spec.claim_id}: mismatch value recipe but value equals official"
                )

        if spec.intended_verdict == "match" and spec.gold_stat_id and (sch := spec.gold_schema):
            tp = sch.time_period or ""
            fact = (spec.gold_stat_id, tp)
            if fact in seen_match_fact:
                errors.append(
                    f"{spec.claim_id}: duplicate match stat+period in article specs"
                )
            seen_match_fact.add(fact)

    region_tokens = sorted(story_region_tokens_from_specs(specs))
    if reject_cross_region and len(region_tokens) >= 2:
        errors.append(f"cross-region specs: {region_tokens}")

    if article_scope == "national" and region_tokens:
        errors.append(f"local region in national article specs: {region_tokens}")

    return len(errors) == 0, errors
