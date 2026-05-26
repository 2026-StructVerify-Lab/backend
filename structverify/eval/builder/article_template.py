"""Deterministic eval article layout tuned for production claim detection."""
from __future__ import annotations

import re
from typing import Any

from structverify.eval.builder.schemas import ClaimSpec
from structverify.eval.builder.story_coherence import StoryAnchor
from structverify.eval.builder.text_utils import claim_text_reflects_gold_value

_DATE_LINE_RE = re.compile(r"^날짜:\s*\d{4}-\d{2}-\d{2}\s*$", re.MULTILINE)

_LEAD_PARAS = (
    "(서울=연합뉴스) 관계 기관이 공식 통계를 바탕으로 보도 자료를 공개했다.",
    "이번 내용은 공공기관이 발표한 조사·통계 자료를 토대로 정리했다.",
)

_CLOSING_PARAS = (
    "관계 기관은 향후에도 동향 분석을 이어갈 계획이다.",
    "전문가들은 공식 통계를 근거로 정책 방향을 점검할 필요가 있다고 말했다.",
)


def ensure_headline_blank_line(body: str) -> str:
    """`# 제목` 다음에 반드시 빈 줄(\\n\\n)이 오도록 보정."""
    text = body.strip()
    if not text.startswith("#"):
        return text
    if re.match(r"^#[^\n]+\n\n", text):
        return text
    fixed = re.sub(r"^(#[^\n]+)\n(?!\n)", r"\1\n\n", text, count=1)
    return fixed


_METADATA_HEADLINE_RE = re.compile(r"(연도=|지역=|조사=)")


def _short_indicator(indicator: str | None) -> str:
    ind = (indicator or "지표").strip()
    if len(ind) <= 48:
        return ind
    if "-" in ind:
        tail = ind.split("-")[-1].strip()
        if tail:
            return tail[:48]
    return ind[:48].rstrip() + "…"


def human_headline_from_specs(domain: str, specs: list[ClaimSpec]) -> str:
    """뉴스 헤드라인 — narrative_hint(연도=…)는 LLM용 메타데이터이므로 쓰지 않음."""
    if not specs:
        return f"{domain} 관련 통계 보도"
    anchor = StoryAnchor.from_spec(specs[0])
    sch = specs[0].gold_schema
    year = str(anchor.anchor_year) if anchor.anchor_year else None
    if not year and sch and sch.time_period and re.match(r"^\d{4}", str(sch.time_period)):
        year = str(sch.time_period)[:4]
    survey = (anchor.survey_segment or "").strip()
    ind = _short_indicator(sch.indicator if sch else None)

    if year and survey and ind:
        return f"{year}년 {ind}…{survey}"
    if year and ind:
        return f"{year}년 {ind} 관련 수치 공개"
    if year and survey:
        return f"{year}년 {survey} 발표"
    if survey:
        return f"{survey} 관련 보도"
    if ind:
        return f"{ind} 관련 보도"
    return f"{domain} 관련 통계 보도"


def _sanitize_headline(candidate: str, domain: str, specs: list[ClaimSpec]) -> str:
    """LLM이 준 제목이 메타데이터 형태면 폐기."""
    c = re.sub(r"^#+\s*", "", candidate).strip()
    c = re.sub(r"\s+", " ", c)
    if not c or _METADATA_HEADLINE_RE.search(c):
        return human_headline_from_specs(domain, specs)
    if len(c) > 80:
        return human_headline_from_specs(domain, specs)
    return c


def _lead_has_spec_values(para: str, specs: list[ClaimSpec]) -> bool:
    for spec in specs:
        if spec.intended_verdict not in ("match", "mismatch"):
            continue
        sch = spec.gold_schema
        if sch and sch.value is not None and claim_text_reflects_gold_value(
            para, float(sch.value), sch.unit
        ):
            return True
    return False


def extract_headline_and_lead_from_llm(
    article_text: str,
    specs: list[ClaimSpec],
    claim_texts: list[str],
    *,
    domain: str = "general",
) -> tuple[str | None, list[str]]:
    """
    structured 모드: LLM 본문에서 제목·무수치 리드만 추출 (claim·수치 문단 제외).
    """
    if not article_text.strip():
        return None, []

    norm_claims = {re.sub(r"\s+", " ", c.strip()) for c in claim_texts if c.strip()}
    headline: str | None = None
    lead_parts: list[str] = []

    for para in [p.strip() for p in article_text.strip().split("\n\n") if p.strip()]:
        if para.startswith("#"):
            headline = _sanitize_headline(para, domain, specs)
            continue
        if _DATE_LINE_RE.match(para):
            continue
        norm_para = re.sub(r"\s+", " ", para)
        if norm_para in norm_claims:
            continue
        if re.search(r"\d", para):
            continue
        if _lead_has_spec_values(para, specs):
            continue
        if len(para) < 12:
            continue
        lead_parts.append(para)

    return headline, lead_parts[:3]


def _date_line(specs: list[ClaimSpec]) -> str:
    years: list[str] = []
    for s in specs:
        sch = s.gold_schema
        if sch and sch.time_period and re.match(r"^\d{4}", str(sch.time_period)):
            years.append(str(sch.time_period)[:4])
    if not years:
        return ""
    return f"날짜: {max(years)}-06-15"


def assemble_template_article(
    domain: str,
    specs: list[ClaimSpec],
    claim_texts: list[str],
    *,
    headline_override: str | None = None,
    lead_paragraphs: list[str] | None = None,
) -> str:
    """
    탐지 친화 레이아웃:
      # 제목 + 빈 줄 + 날짜(선택) + 무수치 리드 + claim 문단(각 1문장) + 무수치 마무리
    """
    claims = [c.strip() for c in claim_texts if c and c.strip()]
    headline = headline_override or human_headline_from_specs(domain, specs)
    date_line = _date_line(specs)
    leads = list(lead_paragraphs) if lead_paragraphs else list(_LEAD_PARAS)
    if not leads:
        leads = list(_LEAD_PARAS)

    blocks: list[str] = [f"# {headline}"]
    if date_line:
        blocks.append(date_line)
    blocks.extend(leads)
    blocks.extend(claims)
    blocks.append(_CLOSING_PARAS[0])
    body = "\n\n".join(blocks)
    return ensure_headline_blank_line(body)


def split_lead_and_claim_paragraphs(
    body: str,
    claim_texts: list[str],
) -> tuple[str, list[str]]:
    """본문을 리드(골든 claim 문단 제외)와 claim 전용 문단으로 분리."""
    norm_claims = {re.sub(r"\s+", " ", c.strip()) for c in claim_texts if c.strip()}
    lead_parts: list[str] = []
    claim_parts: list[str] = []
    for para in [p.strip() for p in body.strip().split("\n\n") if p.strip()]:
        if para.startswith("#"):
            continue
        if _DATE_LINE_RE.match(para):
            continue
        norm_para = re.sub(r"\s+", " ", para)
        if norm_para in norm_claims:
            claim_parts.append(para)
        else:
            lead_parts.append(para)
    return "\n\n".join(lead_parts), claim_parts
