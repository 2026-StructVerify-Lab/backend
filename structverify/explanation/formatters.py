"""[리팩] explainer에 있던 수치·출처 포맷 헬퍼 분리 — _build_prompt에서 사용"""
from __future__ import annotations

from structverify.core.schemas import Claim, Evidence, MismatchType, VerificationResult


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
