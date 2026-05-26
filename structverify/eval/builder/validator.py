"""Validation for generated eval articles."""
from __future__ import annotations

import re

from structverify.eval.builder.schemas import BuildState, EvalArticle
from structverify.eval.builder.story_coherence import (
    article_claim_regions,
    catalog_path_is_national,
)
from structverify.eval.builder.article_template import split_lead_and_claim_paragraphs
from structverify.eval.builder.text_utils import (
    claim_has_banned_phrasing,
    claim_text_reflects_gold_value,
    lead_contains_gold_values,
    prose_has_malformed_number,
    prose_has_markdown_emphasis,
)

_NUMERIC_RE = re.compile(r"\d")
_FORECAST_RE = re.compile(r"전망|예상|예측|목표|할\s*것으로|될\s*것으로")
_NATIONAL_RE = re.compile(r"전국|한국\s*전체|우리나라\s*전체|전\s*국민")
_LOCAL_PATH_RE = re.compile(
    r"지자체|광역시|특별자치|시기본|군기본|시사회|군사회|구기본"
)
_REPORT_HEADLINE_RE = re.compile(
    r"현황\s*분석|주요\s*결과|동향\s*분석|종합\s*보고|실태조사\s*결과|"
    r"조사\s*결과\s*발표|리포트|보고서|연도\s*="
)
_META_LEAD_RE = re.compile(
    r"이러한\s*데이터|이번\s*조사\s*결과|다음과\s*같다|다음\s*과\s*같이"
)
_BOILERPLATE_LEAD_RE = re.compile(
    r"관계\s*기관이\s*공식\s*통계를\s*바탕으로\s*보도\s*자료를\s*공개"
)


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


_HEADLINE_BLANK_RE = re.compile(r"^#[^\n]+\n\n", re.MULTILINE)


class EvalArticleValidator:
    def __init__(
        self,
        min_article_chars: int = 80,
        *,
        reject_cross_region: bool = True,
        require_headline_blank_line: bool = False,
        reject_lead_gold_values: bool = False,
        reject_banned_claim_phrasing: bool = False,
        reject_malformed_numbers: bool = False,
        reject_report_style_headline: bool = False,
        reject_boilerplate_lead: bool = False,
    ):
        self.min_article_chars = min_article_chars
        self.reject_cross_region = reject_cross_region
        self.require_headline_blank_line = require_headline_blank_line
        self.reject_lead_gold_values = reject_lead_gold_values
        self.reject_banned_claim_phrasing = reject_banned_claim_phrasing
        self.reject_malformed_numbers = reject_malformed_numbers
        self.reject_report_style_headline = reject_report_style_headline
        self.reject_boilerplate_lead = reject_boilerplate_lead

    def validate(
        self,
        article: EvalArticle,
        state: BuildState,
        domain_quota: dict[str, int],
        verdict_quota: dict[str, int],
        quota_tolerance: int = 2,
    ) -> tuple[bool, list[str]]:
        errors: list[str] = []

        body = article.article_text.strip()
        if len(body) < self.min_article_chars:
            errors.append("article_text too short")

        if not article.claims:
            errors.append("no claims")

        if body and not body.lstrip().startswith("#"):
            errors.append("article_text should start with # headline (pipeline markdown format)")

        if prose_has_markdown_emphasis(body):
            errors.append("article_text contains markdown emphasis (**/*/`); use plain text")

        if self.require_headline_blank_line and not _HEADLINE_BLANK_RE.match(body):
            errors.append("article_text: # headline must be followed by blank line (\\n\\n)")

        headline_line = ""
        for line in body.splitlines():
            if line.strip().startswith("#"):
                headline_line = line.strip().lstrip("#").strip()
                break
        if self.reject_report_style_headline and headline_line:
            if _REPORT_HEADLINE_RE.search(headline_line):
                errors.append(
                    "article_text: headline reads like a report, not a news title"
                )

        if self.reject_malformed_numbers and prose_has_malformed_number(body):
            errors.append("article_text contains malformed number (e.g. 1,2,3,4)")

        claim_texts_list = [c.claim_text.strip() for c in article.claims if c.claim_text]
        if self.reject_lead_gold_values and claim_texts_list:
            lead_text, _ = split_lead_and_claim_paragraphs(body, claim_texts_list)
            if lead_contains_gold_values(lead_text, article.claims):
                errors.append("lead paragraphs must not contain gold numeric values")
            if self.reject_boilerplate_lead:
                if _META_LEAD_RE.search(lead_text):
                    errors.append("lead uses meta phrasing (e.g. 이러한 데이터는)")
                if _BOILERPLATE_LEAD_RE.search(lead_text):
                    errors.append("lead uses generic boilerplate opening")

        norm_body = _normalize_ws(body)
        seen_claim_text: set[str] = set()
        seen_match_fact: set[tuple[str, str]] = set()

        for claim in article.claims:
            ct = claim.claim_text.strip()
            if not ct:
                errors.append(f"{claim.claim_id}: empty claim_text")
                continue

            norm_ct = _normalize_ws(ct)
            if norm_ct in seen_claim_text:
                errors.append(f"{claim.claim_id}: duplicate claim_text in article")
            seen_claim_text.add(norm_ct)

            if norm_ct and norm_ct not in norm_body:
                errors.append(f"{claim.claim_id}: claim_text not contained in article_text")

            if prose_has_markdown_emphasis(ct):
                errors.append(
                    f"{claim.claim_id}: claim_text contains markdown emphasis (**/*/`)"
                )

            if claim.gold_verdict in ("match", "mismatch"):
                if not _NUMERIC_RE.search(ct):
                    errors.append(f"{claim.claim_id}: claim_text missing numeric")
                if _FORECAST_RE.search(ct):
                    errors.append(
                        f"{claim.claim_id}: forecast wording in verifiable claim"
                    )
                if self.reject_banned_claim_phrasing and claim_has_banned_phrasing(ct):
                    errors.append(
                        f"{claim.claim_id}: banned phrasing in verifiable claim "
                        "(e.g. 할 수 있, 전망)"
                    )
                sch_v = claim.gold_schema
                if sch_v and sch_v.value is not None:
                    if not claim_text_reflects_gold_value(
                        ct, float(sch_v.value), sch_v.unit
                    ):
                        errors.append(
                            f"{claim.claim_id}: claim_text missing gold_schema.value "
                            f"({sch_v.value}{sch_v.unit or ''})"
                        )

            if claim.gold_verdict == "mismatch" and claim.mismatch_recipe == "value":
                sch = claim.gold_schema
                if sch and claim.gold_official_value is not None and sch.value is not None:
                    if abs(sch.value - claim.gold_official_value) < 1e-9:
                        errors.append(f"{claim.claim_id}: mismatch value equals official")

            if claim.gold_verdict == "match" and claim.gold_stat_id and (sch := claim.gold_schema):
                tp = sch.time_period or ""
                fact = (claim.gold_stat_id, tp)
                if fact in seen_match_fact:
                    errors.append(
                        f"{claim.claim_id}: duplicate match on same stat+period in article"
                    )
                seen_match_fact.add(fact)

            if claim.gold_stat_id and (sch := claim.gold_schema):
                tp = sch.time_period or ""
                key = (claim.gold_stat_id, tp)
                if key in state.used_fact_keys():
                    errors.append(f"{claim.claim_id}: duplicate fact {key}")

                # 지역 stat + 전국 표현
                ev_path = (claim.gold_evidence.category_path if claim.gold_evidence else "") or ""
                if ev_path and _LOCAL_PATH_RE.search(ev_path) and _NATIONAL_RE.search(ct):
                    errors.append(
                        f"{claim.claim_id}: national scope wording for local stat"
                    )

                # % 스케일: gold 0.3인데 claim에 30%만 있고 0.3 없음 등
                if sch and sch.unit and "%" in sch.unit and sch.value is not None:
                    v = float(sch.value)
                    if 0 < v < 1 and re.search(r"\d+\s*%", ct) and str(v) not in ct:
                        pct = int(round(v * 100))
                        if str(pct) in ct and str(v) not in ct:
                            errors.append(
                                f"{claim.claim_id}: percent scale mismatch "
                                f"(gold={v}, claim may use {pct}%)"
                            )

        proj_domain = dict(state.domain_counts)
        proj_domain[article.intended_domain] = proj_domain.get(article.intended_domain, 0) + 1
        for domain, limit in domain_quota.items():
            if proj_domain.get(domain, 0) > limit + quota_tolerance:
                errors.append(f"domain quota exceeded: {domain}")

        proj_verdict = dict(state.verdict_counts)
        for c in article.claims:
            proj_verdict[c.gold_verdict] = proj_verdict.get(c.gold_verdict, 0) + 1
        for verdict, limit in verdict_quota.items():
            if proj_verdict.get(verdict, 0) > limit + quota_tolerance:
                errors.append(f"verdict quota exceeded: {verdict}")

        scope = getattr(article, "article_scope", None) or "local"
        if scope == "national":
            for claim in article.claims:
                ev_path = (
                    (claim.gold_evidence.category_path if claim.gold_evidence else "")
                    or ""
                )
                if ev_path and not catalog_path_is_national(ev_path):
                    errors.append(
                        f"{claim.claim_id}: non-national stat in national article"
                    )
        elif self.reject_cross_region:
            regions = article_claim_regions(article.claims)
            fine = [r for r in regions if r.endswith(("시", "군", "구"))]
            check = fine if len(fine) >= 2 else regions
            if len(set(check)) >= 2:
                errors.append(
                    f"cross-region claims in article: {sorted(set(check))}"
                )

        return len(errors) == 0, errors

    @staticmethod
    def register_article_facts(article: EvalArticle, state: BuildState) -> None:
        for claim in article.claims:
            if claim.gold_stat_id and claim.gold_schema and claim.gold_schema.time_period:
                state.register_fact(claim.gold_stat_id, claim.gold_schema.time_period)

    @staticmethod
    def update_counts(article: EvalArticle, state: BuildState) -> None:
        state.articles_written += 1
        state.claims_written += len(article.claims)
        scope = getattr(article, "article_scope", None) or "local"
        state.scope_counts[scope] = state.scope_counts.get(scope, 0) + 1
        state.domain_counts[article.intended_domain] = (
            state.domain_counts.get(article.intended_domain, 0) + 1
        )
        for c in article.claims:
            state.verdict_counts[c.gold_verdict] = (
                state.verdict_counts.get(c.gold_verdict, 0) + 1
            )
