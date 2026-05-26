"""Pydantic models for golden eval dataset artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

VerdictType = Literal["match", "mismatch", "unverifiable"]
ArticleScope = Literal["national", "local"]
UnverifiableRecipe = Literal["U1", "U2", "U3", "U4", "U5"]
MismatchRecipe = Literal["value", "time"]


class GoldSchema(BaseModel):
    indicator: str
    value: float | None = None
    unit: str | None = None
    time_period: str | None = None
    population: str = "전체"


class GoldEvidence(BaseModel):
    stat_name: str | None = None
    category_path: str | None = None
    org_name: str | None = None


class EvalClaim(BaseModel):
    claim_id: str
    claim_text: str = ""
    gold_schema: GoldSchema | None = None
    gold_stat_id: str | None = None
    gold_official_value: float | None = None
    gold_verdict: VerdictType
    gold_evidence: GoldEvidence | None = None
    mismatch_recipe: MismatchRecipe | None = None
    unverifiable_reason: str | None = None
    unverifiable_recipe: UnverifiableRecipe | None = None


class EvalArticleSource(BaseModel):
    builder_version: str = "eval_builder_v1"
    registry_snapshot: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class EvalArticle(BaseModel):
    article_id: str
    intended_domain: str
    article_scope: ArticleScope = "local"
    article_text: str = ""
    source: EvalArticleSource = Field(default_factory=EvalArticleSource)
    claims: list[EvalClaim] = Field(default_factory=list)


class EvalManifest(BaseModel):
    dataset_id: str
    status: Literal["building", "frozen"] = "building"
    mode: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    article_count: int = 0
    claim_count: int = 0
    quotas: dict[str, Any] = Field(default_factory=dict)
    registry_domains: list[str] = Field(default_factory=list)
    used_facts_count: int = 0
    articles_sha256: str | None = None
    builder_config_hash: str | None = None


class ClaimSpec(BaseModel):
    """Internal spec before LLM prose fill."""

    claim_id: str
    intended_verdict: VerdictType
    gold_schema: GoldSchema | None = None
    gold_stat_id: str | None = None
    gold_official_value: float | None = None
    gold_evidence: GoldEvidence | None = None
    mismatch_recipe: MismatchRecipe | None = None
    unverifiable_reason: str | None = None
    unverifiable_recipe: UnverifiableRecipe | None = None
    catalog_row: dict[str, Any] | None = None


class BuildState(BaseModel):
    """Resume state persisted during generation."""

    dataset_id: str
    mode: str
    seed: int
    articles_written: int = 0
    claims_written: int = 0
    domain_counts: dict[str, int] = Field(default_factory=dict)
    verdict_counts: dict[str, int] = Field(default_factory=dict)
    scope_counts: dict[str, int] = Field(default_factory=dict)
    used_facts: list[list[str]] = Field(default_factory=list)
    failed_stat_ids: list[str] = Field(default_factory=list)
    next_article_seq: int = 1

    def used_fact_keys(self) -> set[tuple[str, str]]:
        return {(f[0], f[1]) for f in self.used_facts if len(f) >= 2}

    def failed_stat_id_set(self) -> set[str]:
        return set(self.failed_stat_ids)

    def register_fact(self, stat_id: str, time_period: str) -> None:
        key = [stat_id, time_period]
        if key not in self.used_facts:
            self.used_facts.append(key)

    def register_failed_stat(self, stat_id: str) -> None:
        """fetch probe 실패 stat — 재시도 blacklist (resume 시 build_state에 유지)."""
        if stat_id and stat_id not in self.failed_stat_ids:
            self.failed_stat_ids.append(stat_id)
