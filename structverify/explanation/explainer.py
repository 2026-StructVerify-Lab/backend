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
"""
from __future__ import annotations

from structverify.core.schemas import (
    Claim, Evidence, MismatchType, VerdictType, VerificationResult,
)
from structverify.graph.provenance import render_provenance_text
from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


# ── verdict별 전용 프롬프트 ────────────────────────────────────────────────

MATCH_PROMPT = """당신은 팩트체크 전문 작가입니다.
아래 검증 결과를 독자가 이해하기 쉽게 한국어로 설명하세요.

[판정: ✅ 일치 (MATCH) — 이 판정은 확정입니다. 절대 "사실이 아니다"라고 쓰지 마세요.]

[Claim 정보]
원문: "{claim_text}"
지표: {indicator}
claim value의 의미: {value_role}
claim 시점: {claim_time}
기사 수치: {claimed_value} {unit}

[Evidence 정보 (KOSIS 공식 통계)]
공식 수치: {official_value} {evidence_unit}
공식 시점: {evidence_time}
근거 통계: {stat_source}

[검증 결과]
오차: {diff_pct:.1f}%
신뢰도: {confidence:.0%}
{c2_calculation_block}

[출처]
{provenance}

[작성 규칙]
- 2~3문장으로 간결하게
- 판정이 "일치"이므로 "사실입니다", "확인됩니다" 등 긍정적 표현 사용
- "KOSIS {{통계명}}에 따르면" 형식으로 출처 명시
- 기사 수치와 공식 수치를 나란히 비교 (단위 일치 확인)
- claim 시점({claim_time})과 공식 시점({evidence_time})이 다르면 그 점도 명시
- 통계표 ID 포함
- ⚠️ "사실이 아닙니다", "틀렸습니다" 등 부정 표현 절대 금지

[⚠️ 주의 — 설명 전 반드시 확인]
- 공식 통계 출처({stat_source})가 indicator({indicator})와 
  *같은 국가/지역*의 통계인지 확인하세요.
- 시점({claim_time} vs {evidence_time})이 다르면 
  "같은 연도 데이터가 아님"을 반드시 명시하세요.
"""

MISMATCH_PROMPT = """당신은 팩트체크 전문 작가입니다.
아래 검증 결과를 독자가 이해하기 쉽게 한국어로 설명하세요.

[판정: ⚠️ 불일치 (MISMATCH) — 기사 수치와 공식 수치가 다릅니다.]

[Claim 정보]
원문: "{claim_text}"
지표: {indicator}
★ claim value의 의미: {value_role}
claim 시점: {claim_time}
기사 수치: {claimed_value} {unit}

[Evidence 정보 (KOSIS 공식 통계)]
공식 수치: {official_value} {evidence_unit}
공식 시점: {evidence_time}
근거 통계: {stat_source}

[검증 결과]
차이: {diff} {unit} ({diff_pct:.1f}%)
불일치 유형: {mismatch_reason}
신뢰도: {confidence:.0%}
{c2_calculation_block}

[출처]
{provenance}

[★ 의미 구분 — 매우 중요]
- claim value의 의미가 *"차이"* 또는 *"증가율"*이면, 공식 수치의 *절대값*과 *직접 비교하지 마세요*.
  · 예: claim "차이 0.04명" vs evidence "절대값 0.727명" → "0.04가 사실은 0.727"이라는
    표현은 *틀린 비교*. 차이값(0.04)을 절대값(0.727)과 같은 의미로 다루지 말 것.
- claim 시점({claim_time})과 공식 시점({evidence_time})이 다르면 *그것이 차이의 원인*이라고
  명시. 시점 불일치는 *수치 자체의 오류가 아님*.
- 단위가 다르면 (예: claim "%" vs evidence "명") *직접 비교 불가*임을 명시.

[작성 규칙]
- 3~4문장으로 작성
- 위에 명시된 수치만 사용. 새 수치 만들어내지 말 것.
- 불일치 유형에 맞는 *진짜 원인*만 설명 (시점, 단위, 집단, value 의미 차이 등)
- KOSIS 출처 포함
- ⚠️ value 의미(절대값/차이/증가율)를 *반드시 구분*해서 표현"""

UNVERIFIABLE_PROMPT = """당신은 팩트체크 전문 작가입니다.
아래 검증 결과를 독자가 이해하기 쉽게 한국어로 설명하세요.

[판정: ❔ 검증 불가 (UNVERIFIABLE) — 공식 통계로 확정할 수 없습니다.]

[Claim 정보]
원문: "{claim_text}"
지표: {indicator}
claim value의 의미: {value_role}
claim 시점: {claim_time}
기사 수치: {claimed_value} {unit}

[검증 시도]
검증 불가 이유: {reason}
시도한 통계표: {stat_source} 
시도한 검색어: {search_hint}

[작성 규칙]
- 2~3문장으로 작성
- "사실입니다" 또는 "사실이 아닙니다"라고 단정하지 마세요
- 왜 공식 통계를 찾지 못했는지만 설명 (시점 매칭 실패, 단위 불일치 등)
- 독자가 직접 KOSIS에서 확인할 방법 제시
- 수치를 새로 만들어내지 마세요. 위에 명시된 수치만 사용.
"""


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
    """verdict 유형에 따라 적절한 프롬프트를 생성한다.

    [v6.14] 추가 컨텍스트:
      - indicator (지표명)
      - value_role (절대값/차이/증가율) — schema 기반 추론
      - claim_time, evidence_time (시점 비교)
      - evidence_unit (단위 비교)
      - c2_calculation_block (C2 계산 결과 있으면)
    """

    ev = result.evidence
    schema = claim.schema

    claimed_value = schema.value if schema and schema.value is not None else "N/A"
    unit = schema.unit or "" if schema else ""
    official_value = ev.official_value if ev and ev.official_value is not None else "N/A"
    evidence_unit = (ev.unit if ev and ev.unit else "") or unit
    evidence_time = (ev.time_period if ev and ev.time_period else "") or "(미상)"
    stat_source = _format_stat_source(ev)

    # [v6.14] 컨텍스트 추가
    indicator = (schema.indicator if schema and schema.indicator else "(미상)")
    claim_time = (schema.time_period if schema and schema.time_period else "(미상)")
    value_role = _infer_value_role(schema)
    c2_block = _build_c2_block(schema, result)

    if result.verdict == VerdictType.MATCH:
        diff_pct = _calc_diff_pct(claimed_value, official_value)
        return MATCH_PROMPT.format(
            claim_text=claim.claim_text,
            indicator=indicator,
            value_role=value_role,
            claim_time=claim_time,
            evidence_time=evidence_time,
            claimed_value=claimed_value,
            official_value=official_value,
            unit=unit,
            evidence_unit=evidence_unit,
            diff_pct=diff_pct,
            confidence=result.confidence,
            stat_source=stat_source,
            c2_calculation_block=c2_block,
            provenance=prov_text,
        )

    elif result.verdict == VerdictType.MISMATCH:
        diff_pct = _calc_diff_pct(claimed_value, official_value)
        diff = _calc_diff(claimed_value, official_value)
        mismatch_reason = _mismatch_reason_text(result.mismatch_type)
        return MISMATCH_PROMPT.format(
            claim_text=claim.claim_text,
            indicator=indicator,
            value_role=value_role,
            claim_time=claim_time,
            evidence_time=evidence_time,
            claimed_value=claimed_value,
            official_value=official_value,
            unit=unit,
            evidence_unit=evidence_unit,
            diff=diff,
            diff_pct=diff_pct,
            mismatch_reason=mismatch_reason,
            confidence=result.confidence,
            stat_source=stat_source,
            c2_calculation_block=c2_block,
            provenance=prov_text,
        )

    else:  # UNVERIFIABLE
        reason = _unverifiable_reason(claim, result)
        search_hint = _format_search_hint(claim)
        return UNVERIFIABLE_PROMPT.format(
            claim_text=claim.claim_text,
            indicator=indicator,
            value_role=value_role,
            claim_time=claim_time,
            claimed_value=claimed_value,
            unit=unit,
            reason=reason,
            stat_source=stat_source,
            search_hint=search_hint,
            c2_calculation_block=c2_block,
        )


# ── [v6.14] 컨텍스트 헬퍼 ────────────────────────────────────────────────────

def _infer_value_role(schema) -> str:
    """
    claim value의 의미적 역할 추론 — explainer가 절대값/차이/증가율 구분하도록.

    판단 기준 (우선순위):
      1. unit이 "%/퍼센트/율/비율" → 증가율/비율
      2. indicator에 "차이/증감/증가/감소/변화" 포함 → 차이/변화량
      3. schema에 prev_value가 있음 → 차이 또는 증가율 (unit으로 구분)
      4. 그 외 → 절대값
    """
    if schema is None:
        return "절대값"

    unit_lower = (schema.unit or "").lower()
    indicator_lower = (schema.indicator or "").lower()

    is_ratio_unit = any(k in unit_lower for k in ("%", "퍼센트", "percent", "율", "비율"))
    has_diff_word = any(k in indicator_lower for k in (
        "차이", "차", "증감", "증가", "감소", "변화", "delta", "diff"
    ))
    has_prev = getattr(schema, "prev_value", None) is not None

    if is_ratio_unit:
        return "증가율/비율 (절대값이 *아님*, 두 시점 사이의 *변화 비율*)"
    if has_diff_word or has_prev:
        if is_ratio_unit:
            return "증가율 (두 시점 사이의 *변화 비율*)"
        return "차이/변화량 (절대값이 *아님*, 두 시점 사이의 *수치 차이*)"
    return "절대값 (현재 시점의 *실제 측정값*)"


def _build_c2_block(schema, result) -> str:
    """
    C2 (증가율/차이 자동 계산) 정보를 prompt block으로 포맷.

    schema.prev_value가 있으면 표시. verifier가 계산한 결과(있으면)도 포함.
    """
    if schema is None:
        return ""

    prev_value = getattr(schema, "prev_value", None)
    prev_time = getattr(schema, "prev_time_period", None)
    prev_phrase = getattr(schema, "prev_phrase", None)

    if prev_value is None:
        return ""

    lines = ["[C2 — 비교 기준값 정보]"]
    lines.append(f"기준값(prev): {prev_value}" + (f" ({prev_phrase})" if prev_phrase else ""))
    if prev_time:
        lines.append(f"기준 시점: {prev_time}")

    # verifier가 계산했는지 확인 (raw_response 안에 calculated_from_prev 있을 수 있음)
    ev = result.evidence
    raw = (ev.raw_response if ev and isinstance(ev.raw_response, dict) else {}) or {}
    calculated = raw.get("calculated_from_prev")
    if calculated is not None:
        lines.append(
            f"verifier 계산: 현재={ev.official_value}, prev={prev_value} → "
            f"계산값={calculated:.4f}"
        )
        lines.append(f"이 계산값을 claim value와 비교한 결과가 위의 판정.")

    return "\n".join(lines)


def _mismatch_reason_text(mismatch_type: MismatchType | None) -> str:
    """MismatchType을 독자가 이해할 수 있는 설명 문구로 변환한다."""
    mapping = {
        MismatchType.VALUE:       "단순 수치 오류 — 기사가 공식 수치와 다른 값을 인용",
        MismatchType.TIME_PERIOD: "시점 불일치 — 다른 연도의 통계를 현재 수치처럼 인용",
        MismatchType.POPULATION:  "대상 집단 불일치 — 다른 범위(전체 vs 일부)의 통계를 혼용",
        MismatchType.EXAGGERATION:"과장/축소 — 실제 수치보다 크게 또는 작게 표현",
    }
    return mapping.get(mismatch_type, "수치 불일치")


def _unverifiable_reason(claim: Claim, result: VerificationResult) -> str:
    """검증 불가 이유를 구체적으로 서술한다."""
    if result.evidence is None:
        return "KOSIS에서 관련 통계표를 찾지 못함"
    if result.evidence.official_value is None:
        return "통계표는 찾았으나 해당 시점/대상의 수치가 없음"
    if claim.schema is None or claim.schema.value is None:
        return "기사에서 구체적인 수치를 추출하지 못함"
    return "검증에 필요한 정보가 불충분함"


def _format_stat_source(ev: Evidence | None) -> str:
    """Evidence에서 통계 출처 텍스트를 생성한다."""
    if not ev:
        return "N/A"
    parts = []
    if ev.source_name:
        parts.append(ev.source_name)
    if ev.stat_table_id:
        parts.append(f"표ID: {ev.stat_table_id}")
    if ev.time_period:
        parts.append(f"{ev.time_period} 기준")
    return " | ".join(parts) if parts else "N/A"


def _format_search_hint(claim: Claim) -> str:
    """독자가 직접 검색할 수 있는 키워드를 제안한다."""
    if not claim.schema:
        return claim.claim_text[:30]
    parts = []
    if claim.schema.indicator:
        parts.append(claim.schema.indicator)
    if claim.schema.population:
        parts.append(claim.schema.population)
    if claim.schema.time_period:
        parts.append(claim.schema.time_period)
    return " ".join(parts) if parts else claim.claim_text[:30]


def _calc_diff_pct(claimed: float | str, official: float | str) -> float:
    """차이 비율(%) 계산. 수치가 없으면 0 반환."""
    try:
        c, o = float(claimed), float(official)
        if o == 0:
            return 0.0
        return abs(c - o) / abs(o) * 100
    except (TypeError, ValueError):
        return 0.0


def _calc_diff(claimed: float | str, official: float | str) -> str:
    """실제 차이값 계산. 수치가 없으면 'N/A' 반환."""
    try:
        c, o = float(claimed), float(official)
        diff = c - o
        return f"{diff:+.1f}"
    except (TypeError, ValueError):
        return "N/A"


def _fallback_explanation(claim: Claim, result: VerificationResult) -> str:
    """LLM 실패 시 기본 텍스트로 fallback."""
    verdict_kr = {
        VerdictType.MATCH: "일치",
        VerdictType.MISMATCH: "불일치",
        VerdictType.UNVERIFIABLE: "검증 불가",
    }.get(result.verdict, result.verdict.value)

    base = f'"{claim.claim_text[:40]}..." — 판정: {verdict_kr}'

    if result.verdict == VerdictType.MISMATCH and result.evidence:
        ev = result.evidence
        schema = claim.schema
        if schema and schema.value and ev.official_value:
            base += (
                f" | 기사: {schema.value}{schema.unit or ''}"
                f" / 공식: {ev.official_value}{ev.unit or ''}"
            )

    if result.provenance_summary:
        base += f" | {result.provenance_summary}"

    return base