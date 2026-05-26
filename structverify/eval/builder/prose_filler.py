"""LLM prose generation for eval articles (text only; labels from code/KOSIS)."""
from __future__ import annotations

import json
import re
from typing import Any

from structverify.eval.builder.article_template import (
    assemble_template_article,
    ensure_headline_blank_line,
    extract_headline_and_lead_from_llm,
)
from structverify.eval.builder.schemas import ClaimSpec, EvalArticle, EvalArticleSource, EvalClaim
from structverify.eval.builder.story_coherence import StoryAnchor
from structverify.eval.builder.text_utils import (
    build_claim_text_from_spec,
    claim_text_reflects_gold_value,
    format_value_with_unit,
    strip_llm_markdown_emphasis,
)
from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

# 파이프라인 extractor/sir_builder와 동일한 마크다운-ish 형식
ARTICLE_PROSE_PROMPT = """당신은 한국 통계·경제 뉴스 기자입니다.
아래 스펙에 맞는 뉴스 기사 본문과 각 claim 문장을 작성하세요.

[도메인]
{domain}

[스토리 앵커 — 한 기사 = 한 보도 (이 범위를 벗어나지 말 것)]
{story_anchor}

[지역·범위 힌트 — category_path 기반, 문장에 반드시 반영]
{scope_hints}

[기사에 포함할 claim 스펙 — gold 라벨과 문장 수치/시점을 반드시 일치]
{claims_spec_json}

[작성 규칙 — VerificationPipeline 입력 형식]
1. article_text는 실제 언론사 통계·경제 기사처럼 읽혀야 합니다 (보고서·현황 분석 문서 아님).
   - 첫 줄: `# {{헤드라인}}` — 연합뉴스/뉴시스형 제목 (예: "경제적 이유로 진료 못 받은 이웃 243명…의료패널").
     금지: "~현황 분석", "~주요 결과", "~동향 분석", "~실태조사 결과", "~보고서", "연도=…" 형식.
   - `# 제목` 다음 반드시 빈 줄 한 줄.
   - `날짜: YYYY-MM-DD` 줄은 쓰지 마세요 (뉴스 본문에 날짜 라벨 금지).
   - 빈 줄 후 4~7문단. 문단 사이는 빈 줄(\\n\\n)만 사용.
   - 1문단: 리드(무슨 조사·이슈인지, 수치 없이).
   - 2~5문단: 맥락·배경·의미 + 아래 claim 문장을 문맥에 맞게 끼워 넣기.
   - 마지막 1문단: 짧은 마무리(기관 반응·시사점, 수치·전망 표현 없이).
   - 금지: "이러한 데이터는", "이번 조사 결과는", "다음과 같다" 같은 메타 문장만 있는 리드.
   - 금지: "(서울=연합뉴스) 관계 기관이 공식 통계를…" 같은 상투적 보일러플레이트만 반복.
2. 각 claim_text는 서로 다른 문장이며, 문장 전체가 article_text에 그대로 1회만 포함되어야 합니다.
   claim 문장을 번호 목록으로만 나열하지 말고, 본문 흐름 속에 배치하세요.
3. match/mismatch claim:
   - gold_schema의 value, time_period, indicator, unit을 문장에 반드시 반영.
   - 연도·수치·단위를 스펙과 동일하게 (mismatch는 스펙 value가 의도적 오류).
   - "전망", "예상", "목표", "할 것으로" 등 미래 예측 표현 금지.
   - scope_hint가 지역(시·군·구)이면 "전국", "한국 전체" 등 전국 표현 금지.
   - unit이 `%`이면 0~100 스케일 (0.345를 34.5%로 쓰지 말 것 — 스펙 value 그대로).
4. unverifiable claim이 스펙에 있을 때만: recipe 지침을 따르세요 (전망·예상·순위-only 문장은 쓰지 마세요).
5. gold_schema 수치를 임의로 바꾸지 마세요.
6. 스토리 앵커·category_path에 없는 시·군·구 이름을 새로 만들지 마세요 (예: '증진시', '체육시' 금지).
7. 같은 지표에 서로 모순되는 수치 문장을 나란히 쓰지 마세요.
8. 마크다운 강조 금지: **, *, ` 등으로 수치·단어를 감싸지 마세요. (예: **11.0%** → 11.0%)
   헤드라인 첫 줄 `# 제목`만 허용하고, 본문·claim에는 #·불릿·굵게 표기 없이 일반 한국어 문장만 씁니다.
9. claim 문장과 동일한 수치·연도를 번호 목록(1. 2.)이나 불릿으로 다시 쓰지 마세요.
   리드·본문은 맥락 설명만 하고, 검증 대상 수치 문장은 claim_text에만 넣으세요.
10. claim 문장은 통계 보도체(…은(는) …로 나타났다)가 아니라 뉴스 단정형
    (예: "2024년 전국 ○○은(는) 1,234명으로 집계됐다.")으로 작성하세요.
11. JSON만 출력:

{{
  "article_text": "...",
  "claims": [
    {{"claim_id": "...", "claim_text": "..."}}
  ]
}}"""

VALIDATION_RETRY_APPEND = """

[이전 생성이 검증에 실패했습니다. 아래 항목을 반드시 수정한 뒤 다시 작성하세요]
{validation_errors}
"""

_LOCAL_SCOPE_RE = re.compile(
    r"(지자체|광역시|특별자치|시기본|군기본|시사회|군사회|구기본|"
    r"([가-힣]+시)|([가-힣]+군)|([가-힣]+구))"
)


def _scope_hints_from_specs(
    specs: list[ClaimSpec],
    *,
    article_scope: str = "local",
) -> str:
    if article_scope == "national":
        return (
            "- 전국·국가 단위 통계 보도 — '전국', '한국', '국가 전체' 표현 사용 가능\n"
            "- 특정 시·군·구 고유 지명을 새로 만들지 말 것 (지역 한정 보도 금지)"
        )
    lines: list[str] = []
    for s in specs:
        path = ""
        if s.gold_evidence and s.gold_evidence.category_path:
            path = s.gold_evidence.category_path
        elif s.catalog_row:
            path = str(s.catalog_row.get("category_path") or "")
        if not path:
            continue
        local = bool(_LOCAL_SCOPE_RE.search(path))
        hint = "지역·시군구 조사 — '전국'/'한국 전체' 표현 금지, category_path 지명 반영"
        if local:
            # 경로에서 시·군 이름 추출 시도
            for part in path.split(">"):
                part = part.strip()
                if any(x in part for x in ("시", "군", "구")) and "지자체" not in part:
                    hint = f"조사 범위: {part} — 이 지역명을 문장에 써야 함 ('전국' 금지)"
                    break
        lines.append(f"- {s.claim_id}: {hint} (path: {path[:80]}...)")
    return "\n".join(lines) if lines else "- (전국 통계 가능 — 과장된 '전국'만 피할 것)"


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _ensure_claims_in_article(article_text: str, claim_texts: list[str]) -> str:
    """claim 문장이 본문에 없으면 본문 끝에 문단으로 추가 (validator 통과용)."""
    body = article_text.strip()
    missing: list[str] = []
    norm_body = _normalize_ws(body)
    for ct in claim_texts:
        ct = ct.strip()
        if not ct:
            continue
        if _normalize_ws(ct) not in norm_body:
            missing.append(ct)
    if not missing:
        return body
    extra = "\n\n".join(missing)
    logger.debug(f"Injecting {len(missing)} claim sentence(s) into article_text")
    return f"{body}\n\n{extra}"


class LLMProseFiller:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        llm_cfg = self.config.get("llm", {})
        self.llm = LLMClient(config=llm_cfg)
        prose_cfg = self.config.get("prose", {})
        self.max_attempts = int(prose_cfg.get("max_regenerate_attempts", 2))
        self.claim_text_style = str(prose_cfg.get("claim_text_style", "caption")).strip()
        self.force_deterministic_claim_text = bool(
            prose_cfg.get("force_deterministic_claim_text", False)
        )
        # structured = LLM claim 문장(다양) + 템플릿 본문(무수치 리드); template = 전부 템플릿
        self.article_assembly = str(
            prose_cfg.get("article_assembly", "llm")
        ).strip().lower()
        self.claim_fallback_style = str(
            prose_cfg.get("claim_fallback_style", "news_varied")
        ).strip()

    def _build_claim_text(self, spec: ClaimSpec, *, fallback: bool = False) -> str:
        style = self.claim_fallback_style if fallback else self.claim_text_style
        return build_claim_text_from_spec(
            spec,
            style=style,
            variant_index=None,
        )

    @staticmethod
    def _claims_spec_payload(specs: list[ClaimSpec]) -> str:
        payload: list[dict[str, Any]] = []
        for s in specs:
            path = None
            if s.gold_evidence:
                path = s.gold_evidence.category_path
            elif s.catalog_row:
                path = s.catalog_row.get("category_path")
            item: dict[str, Any] = {
                "claim_id": s.claim_id,
                "gold_verdict": s.intended_verdict,
                "gold_schema": s.gold_schema.model_dump() if s.gold_schema else None,
                "unverifiable_recipe": s.unverifiable_recipe,
                "mismatch_recipe": s.mismatch_recipe,
                "category_path": path,
            }
            sch = s.gold_schema
            if s.intended_verdict in ("match", "mismatch") and sch and sch.value is not None:
                item["required_in_claim_text"] = format_value_with_unit(
                    float(sch.value), sch.unit
                )
                if sch.time_period:
                    item["required_year_in_claim_text"] = str(sch.time_period)[:4]
            payload.append(item)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _story_anchor_line(specs: list[ClaimSpec]) -> str:
        if not specs:
            return "단일 도메인 보도"
        return StoryAnchor.from_spec(specs[0]).narrative_hint()

    def _finalize_claim_texts(
        self,
        specs: list[ClaimSpec],
        text_by_id: dict[str, str],
        *,
        allow_template_correction: bool = False,
    ) -> dict[str, str]:
        """골든 claim_text 확정 — v6: match/mismatch는 항상 결정론적 news 문장."""
        out = dict(text_by_id)
        for spec in specs:
            sch = spec.gold_schema
            if (
                self.force_deterministic_claim_text
                and spec.intended_verdict in ("match", "mismatch")
            ):
                out[spec.claim_id] = self._build_claim_text(spec, fallback=True)
                continue
            ct = (out.get(spec.claim_id) or "").strip()
            if not ct:
                out[spec.claim_id] = self._build_claim_text(spec, fallback=True)
                continue
            if (
                allow_template_correction
                and spec.intended_verdict in ("match", "mismatch")
                and sch
                and sch.value is not None
                and not claim_text_reflects_gold_value(ct, float(sch.value), sch.unit)
            ):
                out[spec.claim_id] = self._build_claim_text(spec, fallback=True)
        return out

    def _fallback_prose(self, domain: str, specs: list[ClaimSpec]) -> dict[str, Any]:
        """LLM 실패 시 최소 뉴스형 본문 (템플릿 보일러플레이트 조립은 v6에서 사용 안 함)."""
        claim_texts = [
            {"claim_id": s.claim_id, "claim_text": self._build_claim_text(s, fallback=True)}
            for s in specs
        ]
        from structverify.eval.builder.article_template import human_headline_from_specs

        headline = human_headline_from_specs(domain, specs)
        lead = (
            f"정부가 발표한 {domain} 관련 공식 통계가 화제다. "
            "현장에서는 발표 내용을 두고 관심이 이어지고 있다."
        )
        blocks = [f"# {headline}", ""] + [lead, ""] + [c["claim_text"] for c in claim_texts]
        article = ensure_headline_blank_line("\n\n".join(blocks))
        return {"article_text": article, "claims": claim_texts}

    def _template_prose(self, domain: str, specs: list[ClaimSpec]) -> dict[str, Any]:
        """LLM 없이 탐지 친화 템플릿 본문 조립."""
        return self._fallback_prose(domain, specs)

    async def fill(
        self,
        article_id: str,
        domain: str,
        specs: list[ClaimSpec],
        registry_snapshot: str = "",
        *,
        article_scope: str = "local",
        validation_errors: list[str] | None = None,
        inject_missing_claims: bool = False,
        allow_template_correction: bool = False,
    ) -> EvalArticle:
        allow_template_correction = False
        if self.article_assembly == "template" and self.force_deterministic_claim_text:
            result = self._template_prose(domain, specs)
            allow_template_correction = True
        else:
            prompt = ARTICLE_PROSE_PROMPT.format(
                domain=domain,
                story_anchor=self._story_anchor_line(specs),
                scope_hints=_scope_hints_from_specs(specs, article_scope=article_scope),
                claims_spec_json=self._claims_spec_payload(specs),
            )
            if validation_errors:
                prompt += VALIDATION_RETRY_APPEND.format(
                    validation_errors="\n".join(f"- {e}" for e in validation_errors)
                )
            result = None
            for attempt in range(self.max_attempts):
                try:
                    result = await self.llm.generate_json(
                        prompt=prompt,
                        system_prompt="한국 뉴스 기자. JSON만 출력.",
                        model_tier="light",
                    )
                    if result and result.get("article_text"):
                        break
                except Exception as e:
                    logger.warning(f"LLM prose attempt {attempt + 1} failed: {e}")

            if not result or not result.get("article_text"):
                logger.warning("LLM prose fallback used")
                result = self._fallback_prose(domain, specs)
                allow_template_correction = True

        text_by_id = {
            c.get("claim_id"): c.get("claim_text", "")
            for c in result.get("claims", [])
            if isinstance(c, dict)
        }
        text_by_id = self._finalize_claim_texts(
            specs,
            text_by_id,
            allow_template_correction=allow_template_correction,
        )
        text_by_id = {
            cid: strip_llm_markdown_emphasis(txt)
            for cid, txt in text_by_id.items()
        }
        claim_sentence_list = [
            text_by_id.get(s.claim_id, "").strip()
            for s in specs
            if text_by_id.get(s.claim_id, "").strip()
        ]

        use_template_body = self.article_assembly in ("template", "structured")
        if use_template_body:
            headline_override = None
            lead_paragraphs: list[str] | None = None
            if self.article_assembly == "structured" and result.get("article_text"):
                headline_override, lead_paragraphs = extract_headline_and_lead_from_llm(
                    strip_llm_markdown_emphasis(result.get("article_text", "")),
                    specs,
                    claim_sentence_list,
                    domain=domain,
                )
                if lead_paragraphs and len(lead_paragraphs) < 1:
                    lead_paragraphs = None
            article_text = assemble_template_article(
                domain,
                specs,
                claim_sentence_list,
                headline_override=headline_override,
                lead_paragraphs=lead_paragraphs,
            )
        else:
            raw_article = ensure_headline_blank_line(
                strip_llm_markdown_emphasis(result.get("article_text", ""))
            )
            article_text = raw_article
            if inject_missing_claims:
                article_text = _ensure_claims_in_article(raw_article, claim_sentence_list)

        eval_claims: list[EvalClaim] = []
        for spec in specs:
            eval_claims.append(
                EvalClaim(
                    claim_id=spec.claim_id,
                    claim_text=text_by_id.get(spec.claim_id, ""),
                    gold_schema=spec.gold_schema,
                    gold_stat_id=spec.gold_stat_id,
                    gold_official_value=spec.gold_official_value,
                    gold_verdict=spec.intended_verdict,
                    gold_evidence=spec.gold_evidence,
                    mismatch_recipe=spec.mismatch_recipe,
                    unverifiable_reason=spec.unverifiable_reason,
                    unverifiable_recipe=spec.unverifiable_recipe,
                )
            )

        return EvalArticle(
            article_id=article_id,
            intended_domain=domain,
            article_scope=(
                article_scope
                if article_scope in ("national", "local")
                else "local"
            ),
            article_text=article_text,
            source=EvalArticleSource(registry_snapshot=registry_snapshot),
            claims=eval_claims,
        )
