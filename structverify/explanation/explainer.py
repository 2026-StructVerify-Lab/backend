"""
explanation/explainer.py — LLM 기반 설명 생성 + Provenance 렌더링 (Step 9)

[김예슬 - 2026-04-22]
- 기존 단일 프롬프트 → verdict 유형별 전용 프롬프트 3종으로 분리
  · MATCH_PROMPT    : 일치 판정 — 어떤 통계가 근거인지 중심으로 설명
  · MISMATCH_PROMPT : 불일치 판정 — 차이 수치, 원인 유형, 독자 주의 안내 포함
  · UNVERIFIABLE_PROMPT : 검증 불가 — 왜 못 찾았는지, 다음 확인 방법 제시
- mismatch_type별 원인 설명 문구 자동 생성 (_mismatch_reason_text)
- _format_evidence(): Evidence 없을 때 안전하게 "N/A" 처리
- _format_schema(): ClaimSchema 요약 텍스트 생성
- generate_explanation() 반환값에 provenance_summary 자동 세팅

[참고] ReAct (Yao et al., ICLR 2023)
  Agent의 최종 Observation 단계에서 판정 근거를 자연어로 설명하는 Step 9

[리팩 2026-06 / refactor/v1/js/explanation]
- verdict별 프롬프트 문자열 → prompts/ 로 분리 (import만 유지, 동작 동일)
- 수치·출처 포맷 헬퍼 → formatters.py 로 분리
- LLM 실패 fallback 문구 → fallback.py 로 분리
"""
from __future__ import annotations

from structverify.core.schemas import (
    Claim, VerdictType, VerificationResult,
)
# [리팩] explainer에 있던 포맷 헬퍼 → formatters.py (로직 동일)
from .formatters import (
    _calc_diff,
    _calc_diff_pct,
    _format_search_hint,
    _format_stat_source,
    _mismatch_reason_text,
    _unverifiable_reason,
)
# [리팩] verdict별 LLM 프롬프트 문자열 → prompts/ (동작 변경 없음)
from .prompts.match import MATCH_PROMPT
from .prompts.mismatch import MISMATCH_PROMPT
from .prompts.multihop import MULTIHOP_PROMPT
from .prompts.unverifiable import UNVERIFIABLE_PROMPT
# [리팩] LLM 실패 시 fallback 문구 → fallback.py (explainer에서 re-export)
from .fallback import _fallback_explanation
from structverify.graph.provenance import render_provenance_text
from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


# ── 메인 함수 ─────────────────────────────────────────────────────────────

async def generate_explanation(
    claim: Claim,
    result: VerificationResult,
    config: dict | None = None,
) -> str:
    """
    검증 결과에 대한 자연어 설명을 생성한다.

    verdict 유형에 따라 다른 프롬프트를 사용:
      MATCH        → MATCH_PROMPT (일치 근거 중심)
      MISMATCH     → MISMATCH_PROMPT (차이 원인 + 독자 주의)
      UNVERIFIABLE → UNVERIFIABLE_PROMPT (왜 못 찾았는지 + 직접 확인 방법)

    Args:
        claim: 검증 대상 주장
        result: 검증 결과 (verdict, evidence, mismatch_type 포함)
        config: 설정 dict

    Returns:
        자연어 설명 문자열
    """
    config = config or {}
    llm = LLMClient(config=config.get("llm", {}))

    # Provenance 텍스트 렌더링
    prov_text = "출처 정보 없음"
    if result.evidence and result.evidence.provenance:
        prov_text = render_provenance_text(result.evidence.provenance)
        result.provenance_summary = prov_text

    prompt = _build_prompt(claim, result, prov_text)

    try:
        explanation = await llm.generate(
            prompt=prompt,
            system_prompt="팩트체크 전문 작가. 명확하고 간결한 한국어로 작성하세요.",
            model_tier="heavy",  # HCX-003 — 설명 품질이 중요
        )
        logger.info(f"[Step 9] 설명 생성 완료: {claim.sent_id} ({result.verdict.value})")
        return explanation

    except Exception as e:
        logger.error(f"설명 생성 실패: {e}")
        # fallback — LLM 없이 기본 텍스트 생성
        return _fallback_explanation(claim, result)


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────

def _build_prompt(
    claim: Claim,
    result: VerificationResult,
    prov_text: str,
) -> str:
    """verdict 유형에 따라 적절한 프롬프트를 생성한다."""

    ev = result.evidence
    schema = claim.schema

    claimed_value = schema.value if schema and schema.value is not None else "N/A"
    unit = schema.unit or "" if schema else ""
    official_value = ev.official_value if ev and ev.official_value is not None else "N/A"
    stat_source = _format_stat_source(ev)

    # [Multi-hop] 멀티홉으로 검증된 파생 주장은 전용 프롬프트
    if getattr(result, "multihop_used", False) and result.multihop_detail:
        d = result.multihop_detail
        verdict_label = {
            VerdictType.MATCH:        "✅ 일치 (MATCH)",
            VerdictType.MISMATCH:     "❌ 불일치 (MISMATCH)",
            VerdictType.UNVERIFIABLE: "❓ 검증 불가",
        }.get(result.verdict, str(result.verdict))
        return MULTIHOP_PROMPT.format(
            verdict_label=verdict_label,
            claim_text=claim.claim_text,
            claimed_ratio=d.get("claimed_ratio", "N/A"),
            computed_ratio=d.get("computed_ratio", "N/A"),
            largest_value=f"{d.get('largest_value', 0):,.0f}",
            smallest_value=f"{d.get('smallest_value', 0):,.0f}",
            confidence=result.confidence,
        )

    if result.verdict == VerdictType.MATCH:
        diff_pct = _calc_diff_pct(claimed_value, official_value)
        # [수정] MATCH_PROMPT 템플릿이 쓰는 {indicator}/{claim_time}/
        # {evidence_time} 플레이스홀더가 format 인자에서 누락돼 KeyError가
        # 나던 버그 수정. claim.schema / evidence에서 값을 채운다.
        _indicator = (schema.indicator if schema and schema.indicator
                      else "지표")
        _claim_time = (schema.time_period if schema and schema.time_period
                       else "N/A")
        _evidence_time = (ev.time_period if ev and ev.time_period
                          else "N/A")
        return MATCH_PROMPT.format(
            claim_text=claim.claim_text,
            claimed_value=claimed_value,
            official_value=official_value,
            unit=unit,
            diff_pct=diff_pct,
            confidence=result.confidence,
            stat_source=stat_source,
            provenance=prov_text,
            indicator=_indicator,
            claim_time=_claim_time,
            evidence_time=_evidence_time,
        )

    elif result.verdict == VerdictType.MISMATCH:
        diff_pct = _calc_diff_pct(claimed_value, official_value)
        diff = _calc_diff(claimed_value, official_value)
        mismatch_reason = _mismatch_reason_text(result.mismatch_type)
        return MISMATCH_PROMPT.format(
            claim_text=claim.claim_text,
            claimed_value=claimed_value,
            official_value=official_value,
            unit=unit,
            diff=diff,
            diff_pct=diff_pct,
            mismatch_reason=mismatch_reason,
            confidence=result.confidence,
            stat_source=stat_source,
            provenance=prov_text,
        )

    else:  # UNVERIFIABLE
        reason = _unverifiable_reason(claim, result)
        search_hint = _format_search_hint(claim)
        return UNVERIFIABLE_PROMPT.format(
            claim_text=claim.claim_text,
            reason=reason,
            stat_source=stat_source,
            search_hint=search_hint,
        )

