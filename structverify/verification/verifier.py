"""
verification/verifier.py — Deterministic Verification Engine (Step 8)

수치 비교는 LLM이 아닌 deterministic engine이 수행 (hallucination 방지).

[신준수]
- 수치 비교 로직 및 불일치 유형 세분화 구현 담당
- TIME_PERIOD / POPULATION / EXAGGERATION 불일치 유형 판별 로직 추가

[김예슬 - 2026-05-06 / v2]
- normalize_value(): 천명 → 명 등 단위 변환 추가
- is_same_unit_type(): 단위 타입 불일치 시 UNVERIFIABLE 처리
- 90% 초과 오차 → UNVERIFIABLE (테이블 매칭 오류 방지)
- factcheck_test.py(박재윤) numeric_check 로직 참고

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
# 수정자: 김예슬
# 수정 날짜: 2026-05-06
# 수정 내용: normalize_value / is_same_unit_type / 90% 임계 추가
from __future__ import annotations

import re

from structverify.core.schemas import (
    Claim, Evidence, VerificationResult, VerdictType, MismatchType)
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


# ── [v2 김예슬] 단위 변환 유틸 ────────────────────────────────────────────────

def normalize_value(value: float, kosis_unit: str) -> float:
    """
    KOSIS 단위 → 기본 단위 변환.
    factcheck_test.py(박재윤) normalize_value() 참고.

    예: 3765.4 천명개월 → 3,765,400 명개월
    """
    u = (kosis_unit or "").lower()
    if "천" in u:
        return value * 1_000
    if "백만" in u or "million" in u:
        return value * 1_000_000
    if "억" in u:
        return value * 100_000_000
    return value


def is_same_unit_type(claim_unit: str, kosis_unit: str) -> bool:
    """
    단위 타입이 같은지 확인 (명 ↔ 개월 혼용 방지).
    factcheck_test.py(박재윤) is_same_unit_type() 참고.

    타입이 다르면 False → UNVERIFIABLE 처리.
    한쪽이 비어있으면 True (판단 보류).
    """
    c = (claim_unit or "").lower().strip()
    k = (kosis_unit or "").lower().strip()

    if not c or not k:
        return True

    _TYPES = {
        "people": ["명", "인구", "가구", "세대", "person"],
        "time":   ["개월", "월", "month", "년", "일", "주"],
        "ratio":  ["%", "퍼센트", "percent", "율", "비율"],
        "money":  ["원", "won", "달러", "dollar", "usd"],
    }

    def _get(u: str) -> str:
        for t, kws in _TYPES.items():
            if any(kw in u for kw in kws):
                return t
        return "unknown"

    ct = _get(c)
    kt = _get(k)
    return (ct == "unknown" or kt == "unknown") or (ct == kt)


# ── 메인 검증 함수 ─────────────────────────────────────────────────────────────

def verify_claim(claim: Claim, evidence: Evidence | None,
                 config: dict | None = None) -> VerificationResult:
    """
    공식 통계와 기사 수치를 비교하여 판정.
    검증 자체는 deterministic — LLM 개입 없음.

    [v2 김예슬] 추가된 로직:
      1) 단위 타입 불일치 → UNVERIFIABLE
      2) normalize_value()로 천명 등 단위 변환 후 비교
      3) diff_pct > 90% → UNVERIFIABLE (테이블 매칭 오류 방지)
    """
    config = config or {}
    tolerance = config.get("verification", {}).get("tolerance_percent", 1.0)

    # evidence 없음
    if evidence is None or evidence.official_value is None:
        return VerificationResult(
            claim_id=claim.claim_id, verdict=VerdictType.UNVERIFIABLE,
            confidence=0.3, evidence=evidence)

    # claim schema 없음
    claimed = claim.schema.value if claim.schema else None
    if claimed is None:
        return VerificationResult(
            claim_id=claim.claim_id, verdict=VerdictType.UNVERIFIABLE,
            confidence=0.2, evidence=evidence)

    # value=0.0 → 수치 미추출 → UNVERIFIABLE
    if claimed == 0.0:
        return VerificationResult(
            claim_id=claim.claim_id, verdict=VerdictType.UNVERIFIABLE,
            confidence=0.2, evidence=evidence)

    kosis_unit = evidence.unit or ""
    claim_unit = (claim.schema.unit or "") if claim.schema else ""

    # [v2] 단위 타입 불일치 → UNVERIFIABLE
    if not is_same_unit_type(claim_unit, kosis_unit):
        logger.info(
            f"단위 타입 불일치: claim={claim_unit!r} kosis={kosis_unit!r} → UNVERIFIABLE"
        )
        return VerificationResult(
            claim_id=claim.claim_id, verdict=VerdictType.UNVERIFIABLE,
            confidence=0.3, evidence=evidence)

    # [v2] 단위 변환 후 비교 (천명 → 명 등)
    official = normalize_value(evidence.official_value, kosis_unit)

    diff_pct = abs(claimed - official) / max(abs(official), 1e-9) * 100

    # [v2] 90% 초과 → 테이블 매칭 오류 의심 → UNVERIFIABLE
    if diff_pct > 90:
        logger.info(
            f"오차 {diff_pct:.1f}% 초과 → 테이블 매칭 오류 의심 → UNVERIFIABLE"
        )
        return VerificationResult(
            claim_id=claim.claim_id, verdict=VerdictType.UNVERIFIABLE,
            confidence=0.3, evidence=evidence)

    if diff_pct <= tolerance:
        verdict = VerdictType.MATCH
        mtype   = None
        conf    = min(0.95, 1.0 - diff_pct / 100)
    else:
        verdict = VerdictType.MISMATCH
        conf    = min(0.9, diff_pct / 100)
        mtype   = _classify_mismatch(claim, evidence, diff_pct, config)

    result = VerificationResult(
        claim_id=claim.claim_id, verdict=verdict, confidence=conf,
        evidence=evidence, mismatch_type=mtype)
    logger.info(f"검증 결과: {verdict.value} (차이: {diff_pct:.2f}%)")
    return result


# ── 불일치 세분화 ──────────────────────────────────────────────────────────────

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
    """
    MISMATCH일 때 세부 유형 분류 (LLM 미사용).

    우선순위:
      1) TIME_PERIOD — 양쪽 시점에서 연도를 뽑을 수 있고 서로 다름
      2) POPULATION  — 기사 집단과 증거 집단이 포함 관계가 아님
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

    # 1) 시점: 연도를 양쪽에서 확보했을 때만 비교
    cy = _primary_year_from_period(schema.time_period)
    ey = _primary_year_from_period(evidence.time_period)
    if cy and ey and cy != ey:
        return MismatchType.TIME_PERIOD

    # 2) 집단: raw_response에서 population 문자열 추출
    raw = evidence.raw_response if isinstance(evidence.raw_response, dict) else {}
    ev_pop = raw.get("population") or raw.get("population_label")
    if isinstance(ev_pop, (list, tuple)):
        ev_pop = " ".join(str(x) for x in ev_pop)
    if schema.population and _population_incompatible(schema.population, ev_pop):
        return MismatchType.POPULATION

    # 3) 과장/축소
    if diff_pct > exaggeration_pct:
        return MismatchType.EXAGGERATION

    return MismatchType.VALUE