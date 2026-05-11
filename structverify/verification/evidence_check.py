"""
verification/evidence_check.py — Evidence ↔ Claim 의미 일치 LLM 검증 (Step 7.5)

[v6.2 김예슬 - 2026-05-08]
KOSIS catalog 검색 + agent 선택이 통과시킨 evidence가 실제로 claim과 의미상
일치하는지 LLM이 1회 binary 분류로 확인.

[배경 — 왜 필요한가]
지금까지의 결과 JSON에서:
  claim:    "작년 연평균기온 14.8도"   (평균기온)
  evidence: DT_2OEEG008 "연평균 기온 변화" 0.64625℃   (기온 변화 폭)
  → "평균기온"과 "기온 변화 폭"은 의미가 다른데 catalog/agent가 통과시킴
  → 이대로 verifier에 보내면 잘못된 mismatch 또는 가짜 match 발생

키워드 겹침(`_is_table_relevant`)만으론 "기온"이 둘 다 들어있어서 통과됨.
의미 차이는 LLM만 잡아낼 수 있음.

[설계]
- runtime_agent의 Step 7(retrieve)과 Step 8(verify) 사이에 1회 호출
- evidence_relevant=False면 evidence를 None으로 만들어 verifier가 unverifiable 처리
- 보수적 운영: LLM 실패 시 통과(True 반환) — 정확도보다 가용성 우선

[설계 원칙 — verifier는 LLM 미사용]
verify_claim()은 deterministic 유지 (hallucination 방지).
의미 검증은 evidence 단계에서 미리 차단.
"""
from __future__ import annotations

from typing import Any

from structverify.core.schemas import Claim, Evidence
from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


_RELEVANCE_PROMPT = """다음 검증 주장(claim)과 KOSIS 통계(evidence)가 의미상 일치하는지 판단하세요.

[주장]
- 지표(indicator): {claim_indicator}
- 대상(population): {claim_population}
- 시점(time_period): {claim_time}
- 단위(unit): {claim_unit}
- 주장 수치(value): {claim_value}

[KOSIS 통계]
- 통계표명: {ev_name}
- 공식 수치(official_value): {ev_value}
- 단위: {ev_unit}
- 시점: {ev_time}

위 KOSIS 통계로 위 주장의 수치를 직접 비교/검증할 수 있는지 판단하세요.

[판단 기준]
- 같은 종류의 지표를 측정하는가?
  · 예: claim="평균기온"  vs  ev="기온 변화 폭" → 다름 (전자는 절대값, 후자는 변화량)
  · 예: claim="연평균기온" vs  ev="연평균 기온" → 같음
  · 예: claim="실업률"     vs  ev="실업자 수"   → 다름 (비율 vs 절대수)
- 같은 단위/스케일/카테고리인가?
- 같은 대상 집단/지역인가? (전국 vs 특정 시도, 청년 vs 전체 등)
- 의미상 1:1 비교가 가능한가?

[보수적 판단]
- 명백히 다른 지표면 false
- 같은 키워드가 있어도 의미가 다르면 false (예: "기온" vs "기온 변화 폭")
- 애매하면 false (false positive보다 false negative가 안전)

JSON으로만 답하세요:
{{
  "is_relevant": true/false,
  "reason": "판단 사유 한 줄"
}}
"""


# [v6.5 추가] combiner=ratio_pct / delta 케이스 전용 prompt
# 이 때 claim.unit은 "%" / "+0.04명" 같은 변화량 단위이고,
# evidence들은 *원본 측정값* (절대 수치)이라 단위가 본질적으로 다름.
# 검증해야 할 것은 "evidence가 측정하는 indicator가 claim의 *기준 measurement* 와 같은가".
_RELEVANCE_PROMPT_COMBINED = """다음 검증 주장(claim)과 KOSIS 통계(evidence)의 관계를 판단하세요.

[주장 — {combiner_label} claim]
- 지표(indicator): {claim_indicator}
- 대상(population): {claim_population}
- 시점(time_period): {claim_time}
- 주장 수치(value): {claim_value} {claim_unit}
- 검증 방식: 두 시점의 measurement(원본 측정값)를 가져와서
  {combiner_explanation}로 계산해서 위 수치와 비교합니다.

[evidence 후보 — anchor]
- 통계표명: {ev_name}
- 공식 수치(official_value): {ev_value} {ev_unit}  (이건 *원본 측정값*)
- 시점: {ev_time}

[판단 기준 — 중요]
이 evidence는 "claim이 표현하는 변화율/차이의 *기준 measurement*"입니다.
단위가 claim과 다른 게 *당연*합니다 (claim="%" or "+0.04명" vs evidence="명" or "도").
*단위 차이를 사유로 false 답하지 마세요*. 우리가 계산해서 변환할 것이기 때문입니다.

진짜로 확인할 것:
1) **evidence의 측정 대상**이 **claim이 말하는 측정값의 변화**와 같은 것을 측정하나?
   · 예: claim="출생아 수가 6.7% 증가" → 기준 measurement는 "출생아 수"
        evidence="시군별 연령별 출산율(1천명당)" → 다름 (출생아 수가 아니라 출산율)
        → false
   · 예: claim="출생아 수가 6.7% 증가" → 기준 measurement는 "출생아 수"
        evidence="출생아수(시도/시/군/구) 명" → 같음 (둘 다 출생아 수)
        → true
   · 예: claim="합계출산율 0.04명 증가" → 기준 measurement는 "합계출산율"
        evidence="합계출산율 명" → 같음 → true
2) **같은 대상 집단/지역**인가? (전국 vs 특정 시도)

[보수적 판단]
- 측정 대상이 다르면 false (예: "출생아 수" vs "출산율")
- 측정 대상은 같은데 대상 집단이 어긋나면 false (전국 vs 시도)
- 측정 대상이 같고 집단도 호환되면 단위 차이는 무시하고 true

JSON으로만 답하세요:
{{
  "is_relevant": true/false,
  "reason": "판단 사유 한 줄 (단위 차이가 아닌 진짜 이유)"
}}
"""


_RELEVANCE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_relevant": {
            "type": "boolean",
            "description": "evidence가 claim의 수치를 직접 검증할 수 있는가",
        },
        "reason": {
            "type": "string",
            "description": "판단 사유 한 줄",
        },
    },
    "required": ["is_relevant", "reason"],
}


async def check_evidence_relevance(
    claim: Claim,
    evidence: Evidence,
    config: dict | None = None,
) -> tuple[bool, str]:
    """
    LLM 1회로 evidence ↔ claim 의미 일치 검증.

    Returns:
        (is_relevant, reason)
        - is_relevant=True  → evidence를 verify_claim에 전달
        - is_relevant=False → evidence를 None으로 만들어 unverifiable 처리

    실패 시:
        (True, "check failed — fallthrough") — 보수적으로 통과 (가용성 우선)
    """
    if not claim.schema or not evidence:
        return True, "schema/evidence 없음 — skip"

    config = config or {}
    llm = LLMClient(config=config.get("llm", {}))

    schema = claim.schema

    # [v6.5 추가] combiner에 따라 다른 prompt 사용
    # direct (measurement): 기존 _RELEVANCE_PROMPT (단위 일치까지 엄격)
    # ratio_pct / delta:    _RELEVANCE_PROMPT_COMBINED (단위 차이 허용, measurement 종류만 검증)
    plan = schema.evidence_plan
    combiner = plan.combiner if plan else "direct"

    if combiner in ("ratio_pct", "delta"):
        combiner_label = "변화율" if combiner == "ratio_pct" else "차이"
        combiner_explanation = {
            "ratio_pct": "(현재 시점 - 비교 시점) / 비교 시점 × 100",
            "delta":     "현재 시점 - 비교 시점",
        }[combiner]
        prompt = _RELEVANCE_PROMPT_COMBINED.format(
            combiner_label=combiner_label,
            combiner_explanation=combiner_explanation,
            claim_indicator=schema.indicator or "?",
            claim_population=schema.population or "?",
            claim_time=schema.time_period or "?",
            claim_unit=schema.unit or "?",
            claim_value=schema.value if schema.value is not None else "?",
            ev_name=evidence.source_name or "?",
            ev_value=(
                evidence.official_value
                if evidence.official_value is not None
                else "?"
            ),
            ev_unit=evidence.unit or "?",
            ev_time=evidence.time_period or "?",
        )
    else:
        # 기존 prompt (measurement/direct) — 변경 없음
        prompt = _RELEVANCE_PROMPT.format(
            claim_indicator=schema.indicator or "?",
            claim_population=schema.population or "?",
            claim_time=schema.time_period or "?",
            claim_unit=schema.unit or "?",
            claim_value=schema.value if schema.value is not None else "?",
            ev_name=evidence.source_name or "?",
            ev_value=(
                evidence.official_value
                if evidence.official_value is not None
                else "?"
            ),
            ev_unit=evidence.unit or "?",
            ev_time=evidence.time_period or "?",
        )

    try:
        result = await llm.generate_structured(
            prompt=prompt,
            schema=_RELEVANCE_OUTPUT_SCHEMA,
            system_prompt=(
                "통계 데이터 의미 분석 전문가. "
                "보수적으로 판단하세요 — 애매하면 false."
            ),
        )
        is_rel = bool(result.get("is_relevant", True))
        reason = result.get("reason", "") or ""
        return is_rel, reason
    except Exception as e:
        logger.warning(
            f"evidence relevance LLM 실패 — 보수적 통과 ({type(e).__name__}): {e}"
        )
        return True, f"check failed — fallthrough ({type(e).__name__})"