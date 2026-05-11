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
주장: "{claim_text}"
기사 수치: {claimed_value} {unit}
공식 수치: {official_value} {unit}
오차: {diff_pct:.1f}%
신뢰도: {confidence:.0%}
근거 통계: {stat_source}
출처: {provenance}

[작성 규칙]
- 2~3문장으로 간결하게
- 판정이 "일치"이므로 "사실입니다", "확인됩니다" 등 긍정적 표현 사용
- "KOSIS {{통계명}}에 따르면" 형식으로 출처 명시
- 기사 수치와 공식 수치를 나란히 비교
- 통계표 ID 포함
- ⚠️ "사실이 아닙니다", "틀렸습니다" 등 부정 표현 절대 금지"""

MISMATCH_PROMPT = """당신은 팩트체크 전문 작가입니다.
아래 검증 결과를 독자가 이해하기 쉽게 한국어로 설명하세요.

[판정: 불일치 (MISMATCH) — 기사 수치와 공식 수치가 다릅니다.]
주장: "{claim_text}"
주장 지표: {claim_indicator} ({claim_population})
기사 수치: {claimed_value} {unit}

근거 통계명: {stat_source}
공식 수치: {official_value} {unit}
차이: {diff} {unit} ({diff_pct:.1f}%)
불일치 유형: {mismatch_reason}
신뢰도: {confidence:.0%}
출처: {provenance}

[중요 — 의미 비교 우선]
- 주장 지표("{claim_indicator}")와 근거 통계명("{stat_source}")이 의미상
  같은 것을 측정하는지 먼저 판단하세요.
- 의미가 다르면(예: "평균기온" vs "기온 변화 폭", "실업률" vs "실업자 수") 단정적
  표현을 쓰지 말고, "직접 비교가 어렵다"거나 "다른 지표라 검증이 부족하다"고
  쓰세요. "실제로는 X로 확인되었다"는 표현 금지.

[작성 규칙]
- 3~4문장
- 기사 수치({claimed_value})와 공식 수치({official_value}) 정확히 인용
- 새로운 수치 생성 금지
- 출처는 통계명을 그대로 인용 ("{stat_source}")"""

UNVERIFIABLE_PROMPT = """당신은 팩트체크 전문 작가입니다.
아래 검증 결과를 독자가 이해하기 쉽게 한국어로 설명하세요.

[판정: 검증 불가 (UNVERIFIABLE) — 공식 통계를 찾지 못했습니다.]
주장: "{claim_text}"
검증 불가 이유: {reason}
시도한 검색어: {search_hint}

[작성 규칙]
- 2~3문장으로 작성
- "사실입니다" 또는 "사실이 아닙니다"라고 단정하지 마세요
- 왜 공식 통계를 찾지 못했는지만 설명
- 독자가 직접 KOSIS에서 확인할 방법 제시
- 수치를 새로 만들어내지 마세요. "100.0°C" 같은 거짓 수치 생성 금지"""


# [v6.3] 멀티-에비던스 (delta/ratio) 전용 프롬프트
MATCH_COMBINED_PROMPT = """당신은 팩트체크 전문 작가입니다.
아래 검증 결과를 독자가 이해하기 쉽게 한국어로 설명하세요.

[판정: ✅ 일치 (MATCH) — 이 판정은 확정입니다.]
주장: "{claim_text}"
기사 수치: {claimed_value} {unit} ({combiner_label})

근거 통계: {stat_source}
시점별 공식 수치:
{endpoints_block}

KOSIS 데이터로 계산한 값: {computed_value} {unit}
오차: {diff_pct:.1f}%
신뢰도: {confidence:.0%}
출처: {provenance}

[작성 규칙]
- 2~3문장
- 두 시점의 공식 수치를 모두 언급한 뒤 계산값과 기사 수치를 비교
- "사실입니다", "확인됩니다" 등 긍정적 표현 사용
- 새로운 수치 생성 금지"""


MISMATCH_COMBINED_PROMPT = """당신은 팩트체크 전문 작가입니다.
아래 검증 결과를 독자가 이해하기 쉽게 한국어로 설명하세요.

[판정: 불일치 (MISMATCH) — 기사가 주장한 {combiner_label}와 KOSIS 계산값이 다릅니다.]
주장: "{claim_text}"
주장 지표: {claim_indicator} ({claim_population})
기사 수치: {claimed_value} {unit}

근거 통계: {stat_source}
시점별 공식 수치:
{endpoints_block}

KOSIS 데이터로 계산한 값: {computed_value} {unit}
차이: {diff} {unit} ({diff_pct:.1f}%)
신뢰도: {confidence:.0%}
출처: {provenance}

[작성 규칙]
- 3~4문장
- 두 시점의 공식 수치를 명시 → 계산값 → 기사 수치 순으로 비교
- 새로운 수치 생성 금지
- 출처 통계명을 그대로 인용"""


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

    # ── [v6.3] 멀티-에비던스 (delta/ratio) 분기 ──────────────────────────
    # combiner가 direct가 아니면 computed_value와 시점별 evidence를 같이 보여줌
    is_combined = (
        result.combiner_used in ("delta", "ratio_pct")
        and result.computed_value is not None
        and len(result.supplementary_evidences) >= 2
    )
    if is_combined and result.verdict in (VerdictType.MATCH, VerdictType.MISMATCH):
        return _build_combined_prompt(claim, result, prov_text)

    # ── 기존 단일 evidence 경로 ────────────────────────────────────────
    official_value = ev.official_value if ev and ev.official_value is not None else "N/A"
    stat_source = _format_stat_source(ev)

    if result.verdict == VerdictType.MATCH:
        diff_pct = _calc_diff_pct(claimed_value, official_value)
        return MATCH_PROMPT.format(
            claim_text=claim.claim_text,
            claimed_value=claimed_value,
            official_value=official_value,
            unit=unit,
            diff_pct=diff_pct,
            confidence=result.confidence,
            stat_source=stat_source,
            provenance=prov_text,
        )

    elif result.verdict == VerdictType.MISMATCH:
        diff_pct = _calc_diff_pct(claimed_value, official_value)
        diff = _calc_diff(claimed_value, official_value)
        mismatch_reason = _mismatch_reason_text(result.mismatch_type)
        # [v6.2] claim 지표/대상 prompt에 추가 → LLM이 evidence와 의미 비교 가능
        claim_indicator = (claim.schema.indicator if claim.schema else None) or "?"
        claim_population = (claim.schema.population if claim.schema else None) or "?"
        return MISMATCH_PROMPT.format(
            claim_text=claim.claim_text,
            claim_indicator=claim_indicator,
            claim_population=claim_population,
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
            search_hint=search_hint,
        )


# [v6.3] 멀티-에비던스용 prompt 빌더
def _build_combined_prompt(
    claim: Claim,
    result: VerificationResult,
    prov_text: str,
) -> str:
    """delta/ratio 검증 결과를 위한 prompt 생성."""
    schema = claim.schema
    claimed_value = schema.value if schema and schema.value is not None else "N/A"
    unit = schema.unit or "" if schema else ""
    computed = result.computed_value if result.computed_value is not None else 0.0

    combiner_label = {
        "delta": "차이값",
        "ratio_pct": "증가율(%)",
    }.get(result.combiner_used or "", result.combiner_used or "")

    # 시점별 공식 수치 블록 생성
    endpoints_lines = []
    for ev in result.supplementary_evidences:
        label = ev.requirement_label or ev.requirement_role or "?"
        time = ev.time_period or "?"
        val = ev.official_value if ev.official_value is not None else "N/A"
        ev_unit = ev.unit or ""
        endpoints_lines.append(f"  - {label} ({time}): {val} {ev_unit}")
    endpoints_block = "\n".join(endpoints_lines) if endpoints_lines else "  (시점별 데이터 없음)"

    # 대표 stat_source — 모든 evidence가 같은 표에서 왔으므로 첫 번째로 충분
    primary_ev = result.evidence or (
        result.supplementary_evidences[0] if result.supplementary_evidences else None
    )
    stat_source = _format_stat_source(primary_ev)

    try:
        claimed_num = float(claimed_value) if claimed_value != "N/A" else 0.0
    except (TypeError, ValueError):
        claimed_num = 0.0
    diff = computed - claimed_num
    denom = max(abs(claimed_num), abs(computed), 1e-6)
    diff_pct = abs(diff) / denom * 100

    if result.verdict == VerdictType.MATCH:
        return MATCH_COMBINED_PROMPT.format(
            claim_text=claim.claim_text,
            claimed_value=claimed_value,
            unit=unit,
            combiner_label=combiner_label,
            stat_source=stat_source,
            endpoints_block=endpoints_block,
            computed_value=f"{computed:+.3f}" if combiner_label == "차이값" else f"{computed:.3f}",
            diff_pct=diff_pct,
            confidence=result.confidence,
            provenance=prov_text,
        )
    else:  # MISMATCH
        claim_indicator = (claim.schema.indicator if claim.schema else None) or "?"
        claim_population = (claim.schema.population if claim.schema else None) or "?"
        return MISMATCH_COMBINED_PROMPT.format(
            claim_text=claim.claim_text,
            claim_indicator=claim_indicator,
            claim_population=claim_population,
            claimed_value=claimed_value,
            unit=unit,
            combiner_label=combiner_label,
            stat_source=stat_source,
            endpoints_block=endpoints_block,
            computed_value=f"{computed:+.3f}" if combiner_label == "차이값" else f"{computed:.3f}",
            diff=f"{diff:+.3f}",
            diff_pct=diff_pct,
            confidence=result.confidence,
            provenance=prov_text,
        )


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