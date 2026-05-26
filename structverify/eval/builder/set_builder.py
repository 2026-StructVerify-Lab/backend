"""EvalSetBuilder orchestrator — quota-driven KOSIS-first golden set generation."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml

from structverify.core.config_loader import load_config
from structverify.detection.domain_classifier import DomainRegistry
from structverify.eval.builder.dataset_writer import DatasetWriter
from structverify.eval.builder.domain_mapping import all_eval_domains
from structverify.eval.builder.gold_builder import GoldBuilder
from structverify.eval.builder.kosis_sampler import KosisRowSampler
from structverify.eval.builder.prose_filler import LLMProseFiller
from structverify.eval.builder.schemas import (
    BuildState,
    ClaimSpec,
    EvalArticle,
    EvalManifest,
)
from structverify.eval.builder.unverifiable_recipes import (
    DETECTION_FRIENDLY_RECIPES,
    allocate_verdict_targets,
    build_unverifiable_spec,
    pick_unverifiable_recipe,
)
from structverify.eval.builder.story_coherence import (
    StoryAnchor,
    catalog_row_compatible,
    catalog_row_is_national,
    pick_viable_national_pool,
    pick_viable_region_pool,
    region_compatible_with_anchor,
    region_token_for_spec,
    spec_compatible_with_anchor,
    specs_single_region,
)
from structverify.eval.builder.detection_preflight import validate_article_detection
from structverify.eval.builder.spec_preflight import validate_specs_before_prose
from structverify.eval.builder.validator import EvalArticleValidator
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

BUILDER_VERSION = "eval_builder_v1"
STATE_PATH = Path("eval/builder/.build_state.json")


def load_eval_builder_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_app_config(eval_cfg: dict[str, Any]) -> dict[str, Any]:
    app_cfg = load_config()
    merged = {**app_cfg, **eval_cfg}
    merged["llm"] = app_cfg.get("llm", {})
    merged["kosis"] = {**app_cfg.get("kosis", {}), **eval_cfg.get("kosis", {})}
    return merged


class EvalSetBuilder:
    def __init__(self, eval_config_path: str | Path = "config/eval_builder.yaml"):
        self.eval_cfg = load_eval_builder_config(eval_config_path)
        self.config = merge_app_config(self.eval_cfg)
        self.mode = self.eval_cfg.get("mode", "pilot")
        self.seed = int(self.eval_cfg.get("seed", 42))
        self.dataset_id = self.eval_cfg.get("dataset_id", "structverify_eval_v1")
        self.rng = random.Random(self.seed)

        quota_block = self.eval_cfg.get("quotas", {}).get(self.mode, {})
        self.target_articles = int(quota_block.get("articles", 40))
        self.verdict_ratios = dict(
            quota_block.get(
                "verdict_claims",
                {"match": 0.45, "mismatch": 0.35, "unverifiable": 0.20},
            )
        )
        uv_cfg = self.eval_cfg.get("unverifiable", {})
        if not bool(uv_cfg.get("enabled", True)):
            uv_share = float(self.verdict_ratios.get("unverifiable", 0) or 0)
            if uv_share > 0:
                self.verdict_ratios["unverifiable"] = 0.0
                self.verdict_ratios["match"] = float(
                    self.verdict_ratios.get("match", 0.45)
                ) + uv_share * 0.55
                self.verdict_ratios["mismatch"] = float(
                    self.verdict_ratios.get("mismatch", 0.35)
                ) + uv_share * 0.45
        allowed = uv_cfg.get("allowed_recipes")
        if allowed is not None:
            self.unverifiable_allowed_recipes: list[str] | None = list(allowed)
        elif bool(uv_cfg.get("detection_friendly_only", False)):
            self.unverifiable_allowed_recipes = list(DETECTION_FRIENDLY_RECIPES)
        else:
            self.unverifiable_allowed_recipes = None
        cpa = self.eval_cfg.get("claims_per_article", {})
        self.claims_min = int(cpa.get("min", 2))
        self.claims_max = int(cpa.get("max", 4))
        self.claims_target = int(cpa.get("target", 3))

        registry_path = self.eval_cfg.get("registry_path", "domain-packs/registry.yaml")
        self.registry = DomainRegistry(registry_path)
        self.registry_domains = all_eval_domains(
            list(self.registry.load().keys()),
            self.eval_cfg.get("exclude_domains", ["general"]),
        )
        self.registry_snapshot = f"{registry_path}"

        out_cfg = self.eval_cfg.get("output", {})
        self.output_dir = Path(out_cfg.get("dir", "eval/datasets"))
        self.flush_every = int(out_cfg.get("flush_every", 5))

        self.sampler = KosisRowSampler(config=self.config, seed=self.seed)
        self.gold_builder = GoldBuilder(config=self.config, seed=self.seed)
        self.prose_filler = LLMProseFiller(config=self.config)
        val_cfg = self.eval_cfg.get("validation", {})
        prose_cfg = self.eval_cfg.get("prose", {})
        min_chars = int(
            prose_cfg.get("min_article_chars")
            or val_cfg.get("min_article_chars", 120)
        )
        self.validator = EvalArticleValidator(
            min_article_chars=min_chars,
            reject_cross_region=bool(
                val_cfg.get("reject_cross_region_articles", True)
            ),
            require_headline_blank_line=bool(
                val_cfg.get("require_headline_blank_line", False)
            ),
            reject_lead_gold_values=bool(
                val_cfg.get("reject_lead_gold_values", False)
            ),
            reject_banned_claim_phrasing=bool(
                val_cfg.get("reject_banned_claim_phrasing", False)
            ),
            reject_malformed_numbers=bool(
                val_cfg.get("reject_malformed_numbers", False)
            ),
            reject_report_style_headline=bool(
                val_cfg.get("reject_report_style_headline", False)
            ),
            reject_boilerplate_lead=bool(
                val_cfg.get("reject_boilerplate_lead", False)
            ),
        )
        self.detection_preflight_enabled = bool(
            val_cfg.get("detection_preflight_enabled", False)
        )
        self.detection_preflight_min_claims = int(
            val_cfg.get("detection_preflight_min_claims", 1)
        )
        self.detection_preflight_min_gold_matches = int(
            val_cfg.get("detection_preflight_min_gold_matches", 1)
        )
        self.quota_tolerance = int(val_cfg.get("quota_tolerance", 2))
        coherence_cfg = self.eval_cfg.get("coherence", {})
        self.story_coherence_enabled = bool(
            coherence_cfg.get("story_anchor_enabled", True)
        )
        self.story_year_slack = int(coherence_cfg.get("year_slack", 1))
        self.single_region_per_article = bool(
            coherence_cfg.get("single_region_per_article", True)
        )
        self.max_stat_attempts_per_article = int(
            coherence_cfg.get("max_stat_attempts_per_article", 3)
        )
        self.max_catalog_rows_per_slot = int(
            coherence_cfg.get("max_catalog_rows_per_slot", 15)
        )
        self.slot_attempt_multiplier = int(
            coherence_cfg.get("slot_attempt_multiplier", 4)
        )
        self.bootstrap_catalog_pool = int(
            coherence_cfg.get("bootstrap_catalog_pool", 40)
        )
        self.scope_strategy = str(
            coherence_cfg.get("scope_strategy", "local_only")
        )
        self.domain_shares = self.eval_cfg.get("domain_shares") or {}
        builder_cfg = self.eval_cfg.get("builder", {})
        self.domain_fallback_after_attempts = int(
            builder_cfg.get("domain_fallback_after_attempts", 0)
        )
        self.domain_fallback_soft_quota = bool(
            builder_cfg.get("domain_fallback_soft_quota", True)
        )
        self.max_attempts_multiplier = int(
            builder_cfg.get("max_attempts_multiplier", 5)
        )
        self.claims_prefer_target_weight = float(
            cpa.get("prefer_target_weight", 0.75)
        )
        self.reject_incomplete_articles = bool(
            val_cfg.get("reject_incomplete_articles", True)
        )
        self.spec_preflight_enabled = bool(
            val_cfg.get("spec_preflight_enabled", True)
        )
        self.max_validation_retries = int(
            prose_cfg.get("max_validation_retries", 3)
        )
        self.writer = DatasetWriter(self.output_dir, self.dataset_id)

    def _target_claim_count(self) -> int:
        return self.target_articles * self.claims_target

    async def plan_quotas(self) -> tuple[dict[str, int], dict[str, int]]:
        try:
            density = await self.sampler.scan_domain_density(self.registry_domains)
        except Exception as e:
            logger.warning(f"Density scan failed ({e}); using equal domain weights")
            density = {d: 1 for d in self.registry_domains}
        domain_quota = self.sampler.allocate_domain_quotas(
            self.registry_domains,
            self.target_articles,
            density,
            domain_shares=self.domain_shares or None,
            domain_articles=self.eval_cfg.get("domain_articles") or None,
        )
        verdict_quota = allocate_verdict_targets(
            self._target_claim_count(),
            self.verdict_ratios,
        )
        return domain_quota, verdict_quota

    def _domain_candidates(
        self,
        domain_quota: dict[str, int],
        state: BuildState,
        *,
        exclude: set[str] | None = None,
        soft_quota: bool = False,
    ) -> list[str]:
        exclude = exclude or set()
        candidates: list[str] = []
        for domain, limit in domain_quota.items():
            if domain in exclude:
                continue
            cap = limit + (self.quota_tolerance if soft_quota else 0)
            if state.domain_counts.get(domain, 0) < cap:
                candidates.append(domain)
        return candidates

    def _pick_domain_weighted(
        self,
        candidates: list[str],
        domain_quota: dict[str, int],
        state: BuildState,
        *,
        soft_quota: bool = False,
    ) -> str:
        weights = [
            max(
                domain_quota[d]
                + (self.quota_tolerance if soft_quota else 0)
                - state.domain_counts.get(d, 0),
                1,
            )
            for d in candidates
        ]
        return self.rng.choices(candidates, weights=weights, k=1)[0]

    def _pick_domain(
        self,
        domain_quota: dict[str, int],
        state: BuildState,
        *,
        exclude: set[str] | None = None,
        soft_quota: bool = False,
    ) -> str | None:
        candidates = self._domain_candidates(
            domain_quota, state, exclude=exclude, soft_quota=soft_quota
        )
        if not candidates:
            return None
        return self._pick_domain_weighted(
            candidates, domain_quota, state, soft_quota=soft_quota
        )

    def _pick_domain_fallback(
        self,
        domain_quota: dict[str, int],
        state: BuildState,
        tried_domains: set[str],
    ) -> str | None:
        """After repeated failures on one article slot, try another domain with slack."""
        domain = self._pick_domain(
            domain_quota, state, exclude=tried_domains, soft_quota=False
        )
        if domain:
            return domain
        if self.domain_fallback_soft_quota:
            domain = self._pick_domain(
                domain_quota, state, exclude=tried_domains, soft_quota=True
            )
            if domain:
                return domain
        return self._pick_domain(
            domain_quota, state, exclude=None, soft_quota=self.domain_fallback_soft_quota
        )

    def _bootstrap_article_scope(
        self,
        pool: list[dict],
        article_id: str,
    ) -> tuple[str, list[dict], StoryAnchor, set[str]] | None:
        """Pick national pool when possible; otherwise single-region local (national_first)."""
        if self.scope_strategy == "national_first":
            national_pool = pick_viable_national_pool(
                pool, self.claims_min, self.rng
            )
            if national_pool:
                bootstrap_row = self.rng.choice(national_pool)
                return (
                    "national",
                    national_pool,
                    StoryAnchor.from_catalog_row(bootstrap_row),
                    set(),
                )
            if self.single_region_per_article:
                picked = pick_viable_region_pool(pool, self.claims_min, self.rng)
                if picked:
                    region, region_rows = picked
                    bootstrap_row = self.rng.choice(region_rows)
                    self.rng.shuffle(region_rows)
                    return (
                        "local",
                        region_rows,
                        StoryAnchor.from_catalog_row(bootstrap_row),
                        {region},
                    )
            logger.warning(
                f"Article {article_id} no viable national or local catalog pool "
                f"(need>={self.claims_min} coherent rows)"
            )
            return None

        if self.single_region_per_article:
            picked = pick_viable_region_pool(pool, self.claims_min, self.rng)
            if picked:
                region, region_rows = picked
                bootstrap_row = self.rng.choice(region_rows)
                self.rng.shuffle(region_rows)
                return (
                    "local",
                    region_rows,
                    StoryAnchor.from_catalog_row(bootstrap_row),
                    {region},
                )
        logger.warning(
            f"Article {article_id} no viable local catalog pool "
            f"(need>={self.claims_min} region rows)"
        )
        return None

    def _pick_verdict(self, verdict_quota: dict[str, int], state: BuildState) -> str | None:
        """부족한 verdict에 가중치를 줘 mismatch quota 미달을 줄인다."""
        candidates = [
            v
            for v, limit in verdict_quota.items()
            if state.verdict_counts.get(v, 0) < limit
        ]
        if not candidates:
            return None
        weights = [
            max(verdict_quota[v] - state.verdict_counts.get(v, 0), 1)
            for v in candidates
        ]
        return self.rng.choices(candidates, weights=weights, k=1)[0]

    def _claims_for_article(self, verdict_quota: dict[str, int], state: BuildState) -> int:
        remaining_verdicts = sum(
            max(0, verdict_quota[v] - state.verdict_counts.get(v, 0))
            for v in verdict_quota
        )
        upper = min(self.claims_max, remaining_verdicts)
        lower = min(self.claims_min, upper)
        if lower <= 0:
            return 0
        preferred = min(self.claims_target, upper)
        if (
            preferred >= lower
            and self.rng.random() < self.claims_prefer_target_weight
        ):
            return preferred
        return self.rng.randint(lower, upper) if upper > lower else lower

    def _bump_article_stat_failure(
        self,
        stat_id: str,
        article_stat_failures: dict[str, int],
        article_excluded_stats: set[str],
    ) -> None:
        if not stat_id:
            return
        article_stat_failures[stat_id] = article_stat_failures.get(stat_id, 0) + 1
        if article_stat_failures[stat_id] >= self.max_stat_attempts_per_article:
            article_excluded_stats.add(stat_id)

    async def _build_claim_spec(
        self,
        claim_id: str,
        domain: str,
        verdict: str,
        state: BuildState,
        article_facts: set[tuple[str, str]] | None = None,
        story_anchor: StoryAnchor | None = None,
        prefer_stat_id: str | None = None,
        article_excluded_stats: set[str] | None = None,
        article_stat_failures: dict[str, int] | None = None,
        locked_regions: set[str] | None = None,
        region_candidate_rows: list[dict] | None = None,
        article_scope: str = "local",
    ) -> ClaimSpec | None:
        article_facts = article_facts or set()
        article_excluded_stats = article_excluded_stats or set()
        article_stat_failures = article_stat_failures or {}
        exclude_ids = state.failed_stat_id_set() | article_excluded_stats
        if verdict == "unverifiable":
            allowed_uv = getattr(self, "unverifiable_allowed_recipes", None)
            recipe = pick_unverifiable_recipe(
                self.rng,
                allowed=allowed_uv,  # type: ignore[arg-type]
            )
            catalog_row = None
            if recipe == "U3":
                candidates = await self.sampler.sample_candidates_for_domain(
                    domain,
                    exclude_facts=state.used_fact_keys(),
                    exclude_stat_ids=exclude_ids,
                )
                for row in candidates[: self.max_catalog_rows_per_slot]:
                    if article_scope == "national" and not catalog_row_is_national(row):
                        continue
                    if not catalog_row_compatible(
                        row,
                        story_anchor,
                        year_slack=self.story_year_slack,
                        single_region=self.single_region_per_article,
                        locked_regions=locked_regions,
                    ):
                        continue
                    catalog_row = row
                    break
            spec = build_unverifiable_spec(
                claim_id, domain, recipe, self.rng, catalog_row=catalog_row
            )
            if self.story_coherence_enabled and not spec_compatible_with_anchor(
                spec,
                story_anchor,
                year_slack=self.story_year_slack,
                single_region=self.single_region_per_article,
                locked_regions=locked_regions,
            ):
                return None
            return spec

        candidates: list[dict] = []
        if region_candidate_rows:
            candidates.extend(region_candidate_rows)
        extra = await self.sampler.sample_candidates_for_domain(
            domain,
            exclude_facts=state.used_fact_keys(),
            exclude_stat_ids=exclude_ids,
        )
        seen_ids = {r.get("stat_id") for r in candidates}
        for row in extra:
            sid = row.get("stat_id")
            if sid and sid in seen_ids:
                continue
            if article_scope == "national" and not catalog_row_is_national(row):
                continue
            if locked_regions and self.story_coherence_enabled:
                path = str(row.get("category_path") or "")
                if not region_compatible_with_anchor(
                    story_anchor,
                    path,
                    single_region=self.single_region_per_article,
                    locked_regions=locked_regions,
                ):
                    continue
            candidates.append(row)
            seen_ids.add(sid)
        if prefer_stat_id and prefer_stat_id not in article_excluded_stats:
            candidates = sorted(
                candidates,
                key=lambda r: 0 if r.get("stat_id") == prefer_stat_id else 1,
            )
        elif prefer_stat_id in article_excluded_stats:
            prefer_stat_id = None
        if self.story_coherence_enabled and story_anchor:
            filtered = [
                r
                for r in candidates
                if catalog_row_compatible(
                    r,
                    story_anchor,
                    year_slack=self.story_year_slack,
                    single_region=self.single_region_per_article,
                    locked_regions=locked_regions,
                )
            ]
            if filtered:
                candidates = filtered

        for row in candidates[: self.max_catalog_rows_per_slot]:
            stat_id = row.get("stat_id", "")
            spec = await self.gold_builder.build_verifiable_spec(
                claim_id, row, verdict
            )
            if spec is None:
                if stat_id:
                    self._bump_article_stat_failure(
                        stat_id, article_stat_failures, article_excluded_stats
                    )
                    before = len(state.failed_stat_ids)
                    state.register_failed_stat(stat_id)
                    if len(state.failed_stat_ids) > before:
                        self.writer.write_build_state(STATE_PATH, state)
                continue
            if self.story_coherence_enabled and not spec_compatible_with_anchor(
                spec,
                story_anchor,
                year_slack=self.story_year_slack,
                single_region=self.single_region_per_article,
                locked_regions=locked_regions,
            ):
                if stat_id:
                    self._bump_article_stat_failure(
                        stat_id, article_stat_failures, article_excluded_stats
                    )
                continue
            if article_scope == "national" and region_token_for_spec(spec):
                if stat_id:
                    self._bump_article_stat_failure(
                        stat_id, article_stat_failures, article_excluded_stats
                    )
                continue
            tp = spec.gold_schema.time_period if spec.gold_schema else ""
            key = (spec.gold_stat_id or "", tp or "")
            if key[0] and key in state.used_fact_keys():
                if stat_id:
                    self._bump_article_stat_failure(
                        stat_id, article_stat_failures, article_excluded_stats
                    )
                continue
            if verdict == "match" and key[0] and key in article_facts:
                if stat_id:
                    self._bump_article_stat_failure(
                        stat_id, article_stat_failures, article_excluded_stats
                    )
                continue
            return spec
        return None

    async def build_one_article(
        self,
        domain: str,
        domain_quota: dict[str, int],
        verdict_quota: dict[str, int],
        state: BuildState,
    ) -> bool:
        n_claims = self._claims_for_article(verdict_quota, state)
        if n_claims <= 0:
            return False

        specs: list[ClaimSpec] = []
        article_facts: set[tuple[str, str]] = set()
        article_excluded_stats: set[str] = set()
        article_stat_failures: dict[str, int] = {}
        story_anchor: StoryAnchor | None = None
        locked_regions: set[str] = set()
        region_candidate_rows: list[dict] | None = None
        article_seq = state.next_article_seq
        article_id = f"eval_{domain}_{article_seq:04d}"
        article_scope = "local"

        if self.story_coherence_enabled:
            exclude_ids = state.failed_stat_id_set() | article_excluded_stats
            pool = await self.sampler.sample_catalog_row(
                domain,
                exclude_stat_ids=exclude_ids,
                limit=self.bootstrap_catalog_pool,
            )
            boot = self._bootstrap_article_scope(pool, article_id)
            if not boot:
                return False
            article_scope, region_candidate_rows, story_anchor, locked_regions = boot

        slot_attempts = 0
        slot_mult = self.slot_attempt_multiplier
        if locked_regions:
            slot_mult = max(slot_mult, 8)
        max_slot_attempts = n_claims * slot_mult
        i = 0
        while len(specs) < n_claims and slot_attempts < max_slot_attempts:
            slot_attempts += 1
            verdict = self._pick_verdict(verdict_quota, state)
            if not verdict:
                break
            i += 1
            claim_id = f"{article_id}_c{i:02d}"
            prefer_stat = (
                story_anchor.stat_id
                if story_anchor
                and story_anchor.stat_id not in article_excluded_stats
                else None
            )
            spec = await self._build_claim_spec(
                claim_id,
                domain,
                verdict,
                state,
                article_facts=article_facts,
                story_anchor=story_anchor if self.story_coherence_enabled else None,
                prefer_stat_id=prefer_stat,
                article_excluded_stats=article_excluded_stats,
                article_stat_failures=article_stat_failures,
                locked_regions=locked_regions if self.story_coherence_enabled else None,
                region_candidate_rows=region_candidate_rows,
                article_scope=article_scope,
            )
            if spec is None:
                logger.debug(f"Skip claim slot {claim_id} — spec/coherence failed")
                continue
            if self.story_coherence_enabled and not specs_single_region(specs, spec):
                logger.debug(
                    f"Skip claim slot {claim_id} — cross-region "
                    f"(locked={sorted(locked_regions)})"
                )
                if spec.gold_stat_id:
                    self._bump_article_stat_failure(
                        spec.gold_stat_id,
                        article_stat_failures,
                        article_excluded_stats,
                    )
                continue
            if self.story_coherence_enabled:
                token = region_token_for_spec(spec)
                if token:
                    locked_regions.add(token)
                if story_anchor is None:
                    story_anchor = StoryAnchor.from_spec(spec)
                elif token and not story_anchor.region_token:
                    story_anchor = StoryAnchor.from_spec(spec)
            elif story_anchor is None:
                story_anchor = StoryAnchor.from_spec(spec)
            if spec.gold_stat_id and spec.gold_schema and spec.gold_schema.time_period:
                article_facts.add(
                    (spec.gold_stat_id, spec.gold_schema.time_period)
                )
            specs.append(spec)

        if len(specs) < self.claims_min:
            logger.warning(
                f"Article {article_id} insufficient claims ({len(specs)} < min)"
            )
            return False
        if self.reject_incomplete_articles and len(specs) < n_claims:
            logger.warning(
                f"Article {article_id} rejected: planned {n_claims} claims, "
                f"got {len(specs)}"
            )
            return False

        if self.spec_preflight_enabled:
            pre_ok, pre_errors = validate_specs_before_prose(
                specs,
                reject_cross_region=self.validator.reject_cross_region,
                article_scope=article_scope,
            )
            if not pre_ok:
                logger.warning(
                    f"Spec preflight failed {article_id}: {pre_errors}"
                )
                return False

        tol = self.quota_tolerance
        article: EvalArticle | None = None
        errors: list[str] = []
        for prose_round in range(self.max_validation_retries):
            is_last = prose_round >= self.max_validation_retries - 1
            article = await self.prose_filler.fill(
                article_id=article_id,
                domain=domain,
                specs=specs,
                registry_snapshot=self.registry_snapshot,
                article_scope=article_scope,
                validation_errors=errors if prose_round > 0 else None,
                inject_missing_claims=is_last,
                allow_template_correction=is_last,
            )
            ok, errors = self.validator.validate(
                article, state, domain_quota, verdict_quota, quota_tolerance=tol
            )
            if ok and self.detection_preflight_enabled:
                det_ok, det_errors = await validate_article_detection(
                    article.article_text,
                    [c.claim_text for c in article.claims],
                    self.config,
                    min_claims=self.detection_preflight_min_claims,
                    min_gold_matches=self.detection_preflight_min_gold_matches,
                )
                if not det_ok:
                    ok = False
                    errors = det_errors
            if ok:
                break
            logger.warning(
                f"Validation failed {article_id} "
                f"(round {prose_round + 1}/{self.max_validation_retries}): {errors}"
            )
        else:
            logger.error(f"Reject article {article_id}: {errors}")
            return False

        assert article is not None
        state.next_article_seq += 1
        self.writer.append_article(article)
        self.validator.register_article_facts(article, state)
        self.validator.update_counts(article, state)
        self.writer.write_build_state(STATE_PATH, state)
        logger.info(
            f"Wrote {article_id} domain={domain} scope={article_scope} "
            f"claims={len(article.claims)} "
            f"progress={state.articles_written}/{self.target_articles} "
            f"scope_counts={state.scope_counts}"
        )
        return True

    async def run(self, *, dry_run: bool = False, resume: bool = False) -> EvalManifest:
        if not dry_run:
            count = await self.sampler.count_catalog_rows()
            if count == 0:
                raise RuntimeError(
                    "kosis_stat_catalog is empty — run kosis_crawler or load catalog first"
                )

        domain_quota, verdict_quota = await self.plan_quotas()
        logger.info(f"Domain quotas: {domain_quota}")
        logger.info(f"Verdict quotas: {verdict_quota}")
        logger.info(f"Scope strategy: {self.scope_strategy}")

        if dry_run:
            return EvalManifest(
                dataset_id=self.dataset_id,
                mode=self.mode,
                article_count=self.target_articles,
                claim_count=self._target_claim_count(),
                quotas={
                    "domain_articles": domain_quota,
                    "domain_shares": dict(self.domain_shares),
                    "verdict_claims": verdict_quota,
                    "scope_strategy": self.scope_strategy,
                },
                registry_domains=self.registry_domains,
            )

        state_path = STATE_PATH if resume else None
        if resume and state_path and state_path.exists():
            state = DatasetWriter.load_build_state(
                state_path, self.dataset_id, self.mode, self.seed
            )
            existing = self.writer.load_articles()
            if existing and state.articles_written == 0:
                for art in existing:
                    EvalArticleValidator.register_article_facts(art, state)
                    EvalArticleValidator.update_counts(art, state)
                state.next_article_seq = len(existing) + 1
        else:
            state = BuildState(
                dataset_id=self.dataset_id,
                mode=self.mode,
                seed=self.seed,
            )
            self.writer.ensure_dirs()
            if not resume and self.writer.articles_path.exists():
                logger.warning(
                    f"Truncating existing articles at {self.writer.articles_path}"
                )
                self.writer.articles_path.write_text("", encoding="utf-8")

        attempts = 0
        max_attempts = self.target_articles * self.max_attempts_multiplier
        slot_tried_domains: set[str] = set()
        slot_consecutive_failures = 0
        while state.articles_written < self.target_articles and attempts < max_attempts:
            attempts += 1
            article_slot = state.next_article_seq
            use_fallback = (
                self.domain_fallback_after_attempts > 0
                and slot_consecutive_failures >= self.domain_fallback_after_attempts
                and bool(slot_tried_domains)
            )
            if use_fallback:
                domain = self._pick_domain_fallback(
                    domain_quota, state, slot_tried_domains
                )
                if domain:
                    logger.warning(
                        f"Article slot {article_slot:04d}: domain fallback "
                        f"-> {domain} (tried={sorted(slot_tried_domains)})"
                    )
                    slot_consecutive_failures = 0
            else:
                domain = self._pick_domain(domain_quota, state)
            if not domain:
                logger.error("No domain with remaining quota")
                break
            ok = await self.build_one_article(
                domain, domain_quota, verdict_quota, state
            )
            if ok:
                slot_consecutive_failures = 0
                slot_tried_domains.clear()
            else:
                slot_consecutive_failures += 1
                slot_tried_domains.add(domain)

        manifest = EvalManifest(
            dataset_id=self.dataset_id,
            mode=self.mode,
            article_count=state.articles_written,
            claim_count=state.claims_written,
            quotas={
                "domain_articles": domain_quota,
                "domain_shares": dict(self.domain_shares),
                "verdict_claims": verdict_quota,
                "verdict_ratios": self.verdict_ratios,
                "scope_strategy": self.scope_strategy,
                "scope_counts": dict(state.scope_counts),
            },
            registry_domains=self.registry_domains,
            used_facts_count=len(state.used_facts),
        )
        status = "frozen" if state.articles_written >= self.target_articles else "building"
        self.writer.write_manifest(manifest, self.eval_cfg, status=status)
        return manifest
