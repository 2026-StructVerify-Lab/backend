"""
verification/verifier.py — Deterministic Verification Engine (Step 8) v3

수치 비교는 LLM이 아닌 deterministic engine이 수행 (hallucination 방지).

[신준수]
- 수치 비교 로직 및 불일치 유형 세분화 구현 담당

[김예슬 - 2026-05-06 / v3]
- factcheck_test.py v7(박재윤) numeric_check 로직 전면 반영
  · normalize_value: 천명개월 예외 처리
  · is_same_unit_type: 천명개월 예외 처리
  · 전체 행 탐색: evidence.raw_response["row"] 전체 순회하여 best match
  · 오차 구간: ≤10% MATCH / 10~30% UNVERIFIABLE / 30~90% MISMATCH / >90% UNVERIFIABLE
  · value=0.0 → UNVERIFIABLE
  · 연도 ±2년 필터

[설계 원칙]
- Step 8은 의도적으로 LLM을 사용하지 않습니다
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


# ── [v2 김예슬] 단위 변환 유틸 (박재윤 v2 추가)────────────────────────────────────────────────


def normalize_value(value: float, kosis_unit: str) -> float:
    """
    KOSIS 단위 → 기본 단위 변환.
    [v3] 천명개월은 실제로 개월 단위 (KOSIS 단위명 오류) → 변환 안 함.
    """
    u = (kosis_unit or "").lower()
    # 천명개월은 KOSIS 단위명 오류 — 실제로는 개월 단위
    if "천명개월" in u:
        return value
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
    [v3] 천명개월은 KOSIS 단위명 오류 → 통과.
    """
    c = (claim_unit or "").lower().strip()
    k = (kosis_unit or "").lower().strip()

    if not c or not k:
        return True

    # 천명개월은 KOSIS 단위명 오류 — 비교 자체를 통과
    if "천명개월" in k:
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

    ct, kt = _get(c), _get(k)
    return (ct == "unknown" or kt == "unknown") or (ct == kt)


# ── KOSIS 행에서 수치 추출 ──────────────────────

def _extract_numeric_values(rows: list[dict]) -> list[dict]:
    """raw_response["row"]에서 수치/단위/기간 추출"""
    values = []
    for row in rows:
        dt = row.get("DT", "")
        unit = row.get("UNIT_NM", "")
        prd = row.get("PRD_DE", "")
        try:
            val = float(str(dt).replace(",", ""))
            values.append({"value": val, "unit": unit, "period": prd, "raw": row})
        except (ValueError, TypeError):
            continue
    return values


def _find_best_match(
    claimed: float,
    claim_unit: str,
    claim_year: str | None,
    kosis_values: list[dict],
) -> tuple[dict | None, float]:
    """
    KOSIS 전체 행에서 claim과 가장 가까운 값 탐색.
    factcheck_test.py v7 numeric_check 로직 그대로.

    Returns:
        (best_match_dict, best_error_rate)
    """
    best_match = None
    best_error = float("inf")

    for kv in kosis_values:
        # 연도 ±2년 필터
        if claim_year and kv["period"]:
            kv_year = kv["period"][:4]
            try:
                if abs(int(claim_year) - int(kv_year)) > 2:
                    continue
            except (ValueError, TypeError):
                pass

        normalized = normalize_value(kv["value"], kv["unit"])
        if normalized == 0:
            continue

        # 단위 타입 불일치 → 스킵
        if not is_same_unit_type(claim_unit, kv["unit"]):
            continue

        error_rate = abs(normalized - claimed) / max(abs(claimed), 1)
        if error_rate < best_error:
            best_error = error_rate
            best_match = {**kv, "normalized": normalized, "error_rate": error_rate}

    return best_match, best_error


# ── 메인 검증 함수 ─────────────────────────────────────────────────────────────

def verify_claim(claim: Claim, evidence: Evidence | None,
                 config: dict | None = None,
                 graph: "ClaimGraph | None" = None) -> VerificationResult:
    """
    공식 통계와 기사 수치를 비교하여 판정 (LLM 미사용).

    [v3] factcheck_test.py v7 로직 전면 반영
    [v6 멀티홉] graph가 있으면 claim의 시점을 그래프에서 resolved된 절대 시점으로
                보정하여 KOSIS row 매칭에 사용. claim.schema.time_period가
                "작년" 같은 상대 표현이어도 그래프 traverse로 2023이 나옴.
    """
    config = config or {}

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

    # value=0.0 → 수치 미추출
    if claimed == 0.0:
        return VerificationResult(
            claim_id=claim.claim_id, verdict=VerdictType.UNVERIFIABLE,
            confidence=0.2, evidence=evidence)

    claim_unit = (claim.schema.unit or "") if claim.schema else ""

    # ── 연도 추출 — [v6] 그래프 우선, 그 다음 schema.time_period ────────
    claim_year = None

    # 1) 그래프 멀티홉 traversal 결과 우선
    if graph is not None:
        resolved = graph.resolve_time_for_claim(claim)
        if resolved:
            m = re.search(r"(\d{4})", resolved)
            if m:
                claim_year = m.group(1)
                logger.debug(f"verifier: 그래프에서 resolved year={claim_year} (from {resolved})")

    # 2) 그래프에서 못 찾으면 schema.time_period에서 추출 (fallback)
    if not claim_year and claim.schema and claim.schema.time_period:
        m = re.search(r"(\d{4})", claim.schema.time_period)
        if m:
            claim_year = m.group(1)

    # ── 전체 행 탐색 (factcheck_test.py v7 핵심) ──────────────────────────
    raw = evidence.raw_response if isinstance(evidence.raw_response, dict) else {}
    rows = raw.get("row", [])
    if isinstance(rows, list) and rows:
        kosis_values = _extract_numeric_values(rows)
        if kosis_values:
            best_match, best_error = _find_best_match(
                claimed, claim_unit, claim_year, kosis_values
            )

            if best_match is None:
                return VerificationResult(
                    claim_id=claim.claim_id, verdict=VerdictType.UNVERIFIABLE,
                    confidence=0.3, evidence=evidence)

            return _verdict_from_error(
                claim, evidence, best_error, best_match, config
            )

    # ── 전체 행 없으면 official_value 단독 비교 (폴백) ────────────────────
    kosis_unit = evidence.unit or ""

    if not is_same_unit_type(claim_unit, kosis_unit):
        logger.info(f"단위 타입 불일치: claim={claim_unit!r} kosis={kosis_unit!r}")
        return VerificationResult(
            claim_id=claim.claim_id, verdict=VerdictType.UNVERIFIABLE,
            confidence=0.3, evidence=evidence)

    official = normalize_value(evidence.official_value, kosis_unit)
    diff_pct = abs(claimed - official) / max(abs(claimed), 1) * 100

    return _verdict_from_error(
        claim, evidence, diff_pct / 100, None, config
    )


def _verdict_from_error(
    claim: Claim,
    evidence: Evidence,
    error_rate: float,
    best_match: dict | None,
    config: dict,
) -> VerificationResult:
    """
    오차율 → 판정 결과.

    [v3] factcheck_test.py v7 구간:
      ≤10%   → MATCH
      10~30% → UNVERIFIABLE (LLM 재판정 구간 — Step 8에서는 판단 보류)
      30~90% → MISMATCH
      >90%   → UNVERIFIABLE (테이블 매칭 오류)
    """
    diff_pct = error_rate * 100

    if error_rate <= 0.10:
        verdict = VerdictType.MATCH
        conf = min(0.95, 1.0 - error_rate)
        mtype = None
        logger.info(f"검증 결과: match (오차: {diff_pct:.1f}%)")

    elif error_rate <= 0.30:
        # 유사하나 확신 없음 — LLM 재판정 구간이지만 Step 8은 LLM 미사용
        verdict = VerdictType.UNVERIFIABLE
        conf = 0.4
        mtype = None
        logger.info(f"검증 결과: unverifiable (오차: {diff_pct:.1f}% — 유사하나 확신 없음)")

    elif error_rate > 0.90:
        # 테이블 매칭 오류
        verdict = VerdictType.UNVERIFIABLE
        conf = 0.3
        mtype = None
        logger.info(f"검증 결과: unverifiable (오차: {diff_pct:.1f}% — 테이블 매칭 오류 의심)")

    else:
        # 30~90% → 불일치
        verdict = VerdictType.MISMATCH
        conf = min(0.9, error_rate)
        mtype = _classify_mismatch(claim, evidence, diff_pct, config)
        logger.info(f"검증 결과: mismatch (오차: {diff_pct:.1f}%)")

    return VerificationResult(
        claim_id=claim.claim_id, verdict=verdict, confidence=conf,
        evidence=evidence, mismatch_type=mtype)


# ── 불일치 세분화 (신준수) ─────────────────────────────────────────────────────

def _primary_year_from_period(text: str | None) -> str | None:
    if not text or not str(text).strip():
        return None
    m = re.search(r"(?:19|20)\d{2}", str(text))
    return m.group(0) if m else None


def _norm_token(s: str | None) -> str:
    if not s:
        return ""
    return " ".join(str(s).split()).lower()


def _population_incompatible(claim_pop: str | None, ev_pop: str | None) -> bool:
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
    MISMATCH 세부 유형 분류 (LLM 미사용).
    우선순위: TIME_PERIOD → POPULATION → EXAGGERATION → VALUE
    """
    vconf = config.get("verification", {}) if config else {}
    exaggeration_pct = float(vconf.get("exaggeration_diff_percent", 20.0))

    schema = claim.schema
    if schema is None:
        return (MismatchType.EXAGGERATION if diff_pct > exaggeration_pct
                else MismatchType.VALUE)

    # 시점
    cy = _primary_year_from_period(schema.time_period)
    ey = _primary_year_from_period(evidence.time_period)
    if cy and ey and cy != ey:
        return MismatchType.TIME_PERIOD

    # 집단
    raw = evidence.raw_response if isinstance(evidence.raw_response, dict) else {}
    ev_pop = raw.get("population") or raw.get("population_label")
    if isinstance(ev_pop, (list, tuple)):
        ev_pop = " ".join(str(x) for x in ev_pop)
    if schema.population and _population_incompatible(schema.population, ev_pop):
        return MismatchType.POPULATION

    # 과장
    if diff_pct > exaggeration_pct:
        return MismatchType.EXAGGERATION

    return MismatchType.VALUE