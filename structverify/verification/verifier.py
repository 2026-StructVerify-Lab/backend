"""
verification/verifier.py — Deterministic Verification Engine (Step 8)

수치 비교는 LLM이 아닌 deterministic engine이 수행 (hallucination 방지).

[신준수]
- 수치 비교 로직 및 불일치 유형 세분화 구현 담당
- TIME_PERIOD / POPULATION / EXAGGERATION 불일치 유형 판별 로직 추가

[설계 원칙]
- Step 8은 의도적으로 LLM을 사용하지 않습니다.
- 수치 비교에 LLM을 쓰면 hallucination이 발생할 수 있음 → deterministic만 사용
- 자연어 설명은 Step 9(explainer.py)에서 LLM이 생성

[참고] FEVER (Thorne et al., NAACL 2018)
  SUPPORTS/REFUTES/NEI 3단계 판정 → match/mismatch/unverifiable 매핑
"""
# 수정자: 신준수
# 수정 날짜: 2026-04-27
# 수정 내용: _classify_mismatch 우선순위 분기 및 헬퍼(연도·집단·과장 임계) 구현
from __future__ import annotations

import re

from structverify.core.schemas import (
    Claim, Evidence, VerificationResult, VerdictType, MismatchType)
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


def verify_claim(claim: Claim, evidence: Evidence | None,
                 config: dict | None = None) -> VerificationResult:
    """
    공식 통계와 기사 수치를 비교하여 판정.
    검증 자체는 deterministic — LLM 개입 없음.
    """
    config = config or {}
    tolerance = config.get("verification", {}).get("tolerance_percent", 1.0)

    if evidence is None or evidence.official_value is None:
        return VerificationResult(
            claim_id=claim.claim_id, verdict=VerdictType.UNVERIFIABLE, confidence=0.3,
            evidence=evidence)

    claimed = claim.schema.value if claim.schema else None
    if claimed is None:
        return VerificationResult(
            claim_id=claim.claim_id, verdict=VerdictType.UNVERIFIABLE, confidence=0.2,
            evidence=evidence)

    official = evidence.official_value
    diff_pct = abs(claimed - official) / max(abs(official), 1e-9) * 100

    if diff_pct <= tolerance:
        verdict, mtype, conf = VerdictType.MATCH, None, min(0.95, 1.0 - diff_pct / 100)
    else:
        verdict, conf = VerdictType.MISMATCH, min(0.9, diff_pct / 100)
        # 목적: 상대 오차·config 기반으로 불일치 세부 유형 분류
        mtype = _classify_mismatch(claim, evidence, diff_pct, config)

    result = VerificationResult(
        claim_id=claim.claim_id, verdict=verdict, confidence=conf,
        evidence=evidence, mismatch_type=mtype)
    logger.info(f"검증 결과: {verdict.value} (차이: {diff_pct:.2f}%)")
    return result


def _primary_year_from_period(text: str | None) -> str | None:
    """
    시점 문자열에서 대표 연도(4자리) 하나만 추출.
    둘 다 추출 가능할 때만 TIME_PERIOD 비교에 사용한다.
    """
    if not text or not str(text).strip():
        return None
    m = re.search(r"(?:19|20)\d{2}", str(text))
    return m.group(0) if m else None


def _norm_token(s: str | None) -> str:
    """집단 문자열 비교용: 공백 정리 + 소문자."""
    if not s:
        return ""
    return " ".join(str(s).split()).lower()


def _population_incompatible(claim_pop: str | None, ev_pop: str | None) -> bool:
    """
    기사 집단 vs 증거 집단 설명이 서로 포함 관계가 아니면 '집단 불일치' 후보.
    한쪽만 비어 있으면 판단 보류(False).
    """
    c = _norm_token(claim_pop)
    e = _norm_token(ev_pop)
    if not c or not e:
        return False
    if c in e or e in c:
        return False
    return True


def _classify_mismatch(
  
    claim: Claim, evidence: Evidence, diff_pct: float, config: dict,
) -> MismatchType:
    # v0
    # """
    #    구현 예시:
    #     if claim.schema and evidence:
    #         # 시점 불일치 체크
    #         if (claim.schema.time_period and evidence.time_period and
    #                 claim.schema.time_period != evidence.time_period):
    #             return MismatchType.TIME_PERIOD
    #         # 대상 불일치 체크
    #         if (claim.schema.population and
    #                 claim.schema.population not in (evidence.raw_response.get("population", "") or "")):
    #             return MismatchType.POPULATION
    #         # 과장 체크
    #         if claim.schema.value and evidence.official_value:
    #             diff = abs(claim.schema.value - evidence.official_value)
    #             if diff / max(abs(evidence.official_value), 1e-9) > 0.2:
    #                 return MismatchType.EXAGGERATION
    #     return MismatchType.VALUE
    # """
    # # TODO [신준수]: 위 로직 구현
    """
    MISMATCH일 때 세부 유형 분류 (LLM 미사용).

    우선순위:
      1) TIME_PERIOD — 양쪽 시점에서 연도를 뽑을 수 있고 서로 다름
      2) POPULATION — raw_response['population'] 등으로 증거 집단을 알 수 있고
                      기사 집단과 포함 관계가 아님
      3) EXAGGERATION — 상대 오차가 exaggeration_diff_percent(기본 20%) 초과
      4) VALUE — 위에 해당 없으면 단순 수치 오차
    """

    vconf = config.get("verification", {}) if config else {}
    exaggeration_pct = float(vconf.get("exaggeration_diff_percent", 20.0))

    schema = claim.schema
    if schema is None:
        return (
            MismatchType.EXAGGERATION if diff_pct > exaggeration_pct
            else MismatchType.VALUE
        )

    # --- 1) 시점: 연도를 양쪽에서 확보했을 때만 비교 (한쪽만 있으면 스킵) ---
    cy = _primary_year_from_period(schema.time_period)
    ey = _primary_year_from_period(evidence.time_period)
    if cy and ey and cy != ey:
        return MismatchType.TIME_PERIOD

    # --- 2) 집단: Step 7에서 raw_response 등에 실어준 문자열만 신뢰 ---
    raw = evidence.raw_response if isinstance(evidence.raw_response, dict) else {}
    ev_pop = raw.get("population")
    if ev_pop is None:
        ev_pop = raw.get("population_label")
    if isinstance(ev_pop, (list, tuple)):
        ev_pop = " ".join(str(x) for x in ev_pop)
    if schema.population and _population_incompatible(schema.population, ev_pop):
        return MismatchType.POPULATION

    # --- 3) 과장/축소: 수치 괴리가 매우 큰 경우 ---
    if diff_pct > exaggeration_pct:
        return MismatchType.EXAGGERATION

    return MismatchType.VALUE
