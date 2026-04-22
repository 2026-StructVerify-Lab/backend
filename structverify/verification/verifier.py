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
from __future__ import annotations
import math
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
        mtype = _classify_mismatch(claim, evidence)

    result = VerificationResult(
        claim_id=claim.claim_id, verdict=verdict, confidence=conf,
        evidence=evidence, mismatch_type=mtype)
    logger.info(f"검증 결과: {verdict.value} (차이: {diff_pct:.2f}%)")
    return result


def _classify_mismatch(claim: Claim, evidence: Evidence) -> MismatchType:
    """
    불일치 유형 세분화 분류기.

    TODO [신준수]: 불일치 유형별 판별 로직 구현
      현재: 모두 VALUE로 반환 (placeholder)

      개선 계획:

      [TIME_PERIOD] 시점 불일치
        - claim.schema.time_period != evidence.time_period 이고
          수치 자체는 다른 연도에 맞는 값인 경우
        - 예: 기사는 "2023년 64.2%"인데 공식통계는 2022년 값 62.1%

      [POPULATION] 대상 집단 불일치
        - claim.schema.population과 evidence의 대상이 다른 경우
        - 예: 기사는 "과수 농가"인데 조회된 통계는 "전체 농가"

      [EXAGGERATION] 과장/축소
        - diff_pct가 매우 크거나 (>20%), 방향이 반대인 경우
        - 예: 실제 증가율 3%인데 기사는 "30% 급증" 주장

      구현 예시:
        if claim.schema and evidence:
            # 시점 불일치 체크
            if (claim.schema.time_period and evidence.time_period and
                    claim.schema.time_period != evidence.time_period):
                return MismatchType.TIME_PERIOD
            # 대상 불일치 체크
            if (claim.schema.population and
                    claim.schema.population not in (evidence.raw_response.get("population", "") or "")):
                return MismatchType.POPULATION
            # 과장 체크
            if claim.schema.value and evidence.official_value:
                diff = abs(claim.schema.value - evidence.official_value)
                if diff / max(abs(evidence.official_value), 1e-9) > 0.2:
                    return MismatchType.EXAGGERATION
        return MismatchType.VALUE
    """
    # TODO [신준수]: 위 로직 구현
    return MismatchType.VALUE
