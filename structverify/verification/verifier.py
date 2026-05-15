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


  # 수정자: 박재윤
# 수정 날짜: 2026-05-14
# 수정 내용: 연도 필터 ±2 → 정확 일치로 변경
#   · 기존: abs(claim_year - kv_year) > 2 → 다른 연도 데이터 비교 허용
#   · 변경: abs(claim_year - kv_year) > 0 → 연도 불일치 시 무조건 skip
#   · 이유: 2026년 기사 수치를 2024년 KOSIS 연간 평균과 비교하는 오판정 방지
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


def is_same_unit_type(
    claim_unit: str,
    kosis_unit: str,
    all_rows_empty: bool = False,
) -> bool:
    """
    단위 타입이 같은지 확인 (명 ↔ 개월 혼용 방지).
    [v3] 천명개월은 KOSIS 단위명 오류 → 통과.
    [v6.14] 비대칭 처리:
      - claim_unit 비어있으면 → True (claim 측 정보 부족, 책임은 claim에)
      - kosis_unit 비어있으면 → False (KOSIS row 단위 없으면 안전 차단)
      이전: 둘 중 하나라도 비면 무조건 True → 임의 매칭 발생 (false match 4건)
    [v6.14 G fix] all_rows_empty 파라미터 추가:
      - all_rows_empty=True (표의 *모든* row.UNIT_NM이 빈 칸) → 지표명-단위 일체형 표
        (합계출산율/사망률/혼인 건수 등) → 통과 (관대)
      - all_rows_empty=False (일부 row만 빈 칸) → 그 row는 별도 분류 가능성 → 차단 (안전)
      합계출산율 0.76명, 혼인 건수 17921건 등이 차단되던 문제 해결.
    """
    c = (claim_unit or "").lower().strip()
    k = (kosis_unit or "").lower().strip()

    # [v6.14] claim 측 단위가 없는 경우 → 통과 (claim 책임)
    if not c:
        return True

    # [v6.14] KOSIS row 단위가 없는 경우
    if not k:
        # [v6.14 G fix] 표 전체가 unit 빈 칸이면 (지표명-단위 일체형 표) → 통과
        # 일부만 빈 칸이면 (별도 분류 가능성) → 차단
        return all_rows_empty

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
    claim_year_month: str | None = None,
) -> tuple[dict | None, float]:
    """
    KOSIS 전체 행에서 claim과 가장 가까운 값 탐색.
    factcheck_test.py v7 numeric_check 로직 그대로.

    [v6.14] 진단 로깅 추가:
      - 후보 row 분포 (전체/단위필터 후/연도필터 후)
      - 최종 best_match의 (period, unit, value, error_rate) 출력

    [v6.14 G fix] 단위 처리 개선:
      - 표 전체의 row.UNIT_NM이 비어있는지 사전 분석 (all_rows_empty)
      - all_rows_empty=True (지표명-단위 일체형 표, 예: 합계출산율) → row.unit 비어도 통과
      - all_rows_empty=False (일부만 빈 칸) → 그 row 차단 (안전)

    [v6.14 F1 fix] 시점 정확 일치 우선 매칭:
      - claim_year_month 있으면 (예: "202504") 동일 연-월 row를 *최우선* picking
      - 동일 연-월 없으면 → 동일 연도 row 우선
      - 그것도 없으면 → ±2년 안 모든 row (기존)
      이전 동작: 시점 무시하고 *수치 거리 최소* row picking → 4월 claim이 1월 row와
                매칭되는 등 시점 부정확 발생.

    Returns:
        (best_match_dict, best_error_rate)
    """
    # [v6.14 G fix] 표 전체 row.unit 분포 분석
    nonempty_units = [
        kv["unit"] for kv in kosis_values
        if kv.get("unit") and str(kv["unit"]).strip()
    ]
    all_rows_empty = len(nonempty_units) == 0

    # [v6.14] 진단용 카운터
    total_rows = len(kosis_values)
    year_filtered = 0
    unit_filtered = 0
    zero_filtered = 0

    # [v6.14 F1] 시점 tier별 후보 분류
    # tier 1: 동일 연-월 (가장 정확)
    # tier 2: 동일 연도 (월 다름)
    # tier 3: ±2년 안 다른 연도
    tier1_candidates: list[dict] = []  # 동일 연-월
    tier2_candidates: list[dict] = []  # 동일 연도
    tier3_candidates: list[dict] = []  # ±2년

    for kv in kosis_values:
        # 연도 ±2년 필터
        kv_year = None
        kv_period = kv.get("period") or ""
        if claim_year and kv_period:
            kv_year = kv_period[:4]
            try:
                # · 연도 정확 일치 필터 (박재윤 2026-05-14: ±2 → 0으로 변경)
                if abs(int(claim_year) - int(kv_year)) > 0:
                    year_filtered += 1
                    continue
            except (ValueError, TypeError):
                pass

        normalized = normalize_value(kv["value"], kv["unit"])
        if normalized == 0:
            zero_filtered += 1
            continue

        # 단위 타입 불일치 → 스킵
        if not is_same_unit_type(claim_unit, kv["unit"], all_rows_empty=all_rows_empty):
            unit_filtered += 1
            continue

        # [v6.14 H fix] 상대 오차 공식 수정 — 분모 1 버그 해결
        # 이전: max(abs(claimed), 1) → claimed가 1보다 작으면 분모=1 고정,
        #       소수 지표(합계출산율 0.7대, 비율 0.x 등)에서 오차율이 *심하게 축소*됨.
        # 예: claim=0.04, normalized=0.676 → 이전 63.6% (MISMATCH),
        #     정정 후 94.1% (UNVERIFIABLE) — 실제로는 *15.9배 차이*인데 mismatch로 잘못 판정되던 것.
        # 새 공식: 둘 중 큰 절대값 기준 + epsilon (0 나누기 방지).
        denom = max(abs(normalized), abs(claimed), 1e-9)
        error_rate = abs(normalized - claimed) / denom
        kv_with_meta = {**kv, "normalized": normalized, "error_rate": error_rate}

        # [F1] 시점 tier 분류
        if claim_year_month and kv_period:
            # KOSIS PRD_DE 형식: "YYYYMM" 또는 "YYYY"
            if kv_period.startswith(claim_year_month):
                tier1_candidates.append(kv_with_meta)
            elif claim_year and kv_period.startswith(claim_year):
                tier2_candidates.append(kv_with_meta)
            else:
                tier3_candidates.append(kv_with_meta)
        elif claim_year and kv_period and kv_period.startswith(claim_year):
            tier2_candidates.append(kv_with_meta)
        else:
            tier3_candidates.append(kv_with_meta)

    # [F1] 가장 높은 tier에서 picking
    selected_tier = None
    pool: list[dict] = []
    if tier1_candidates:
        pool = tier1_candidates
        selected_tier = "1 (동일 연-월)"
    elif tier2_candidates:
        pool = tier2_candidates
        selected_tier = "2 (동일 연도)"
    elif tier3_candidates:
        pool = tier3_candidates
        selected_tier = "3 (±2년)"

    best_match = None
    best_error = float("inf")
    for kv in pool:
        if kv["error_rate"] < best_error:
            best_error = kv["error_rate"]
            best_match = kv

    candidates = len(pool)

    # [v6.14] 진단 로깅
    logger.info(
        f"[verifier] match 탐색: claim={claimed}/{claim_unit!r} year={claim_year} "
        f"ym={claim_year_month} | 전체 row={total_rows} (all_rows_empty={all_rows_empty}) → "
        f"연도제외={year_filtered}, zero제외={zero_filtered}, 단위불일치제외={unit_filtered} | "
        f"tier1(연-월)={len(tier1_candidates)}, tier2(연도)={len(tier2_candidates)}, "
        f"tier3(±2년)={len(tier3_candidates)} → 선택 tier={selected_tier}, 최종 후보={candidates}"
    )
    if best_match:
        logger.info(
            f"[verifier] best_match: period={best_match.get('period')!r} "
            f"unit={best_match.get('unit')!r} value={best_match.get('value')} "
            f"(normalized={best_match.get('normalized')}) "
            f"error={best_match.get('error_rate'):.4f}"
        )
    else:
        logger.info(f"[verifier] best_match: 없음 (단위/연도 필터 통과 row 없음)")

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
    claim_year_month = None  # [v6.14 F1] 동일 연-월 우선 매칭용 (예: "202504")

    # 1) 그래프 멀티홉 traversal 결과 우선
    if graph is not None:
        resolved = graph.resolve_time_for_claim(claim)
        if resolved:
            m = re.search(r"(\d{4})", resolved)
            if m:
                claim_year = m.group(1)
                logger.debug(f"verifier: 그래프에서 resolved year={claim_year} (from {resolved})")
            # [F1] 연-월도 추출 ("2025-04" → "202504")
            ym = re.search(r"(\d{4})[-/]?(\d{2})", resolved)
            if ym:
                claim_year_month = ym.group(1) + ym.group(2)

    # 2) 그래프에서 못 찾으면 schema.time_period에서 추출 (fallback)
    if not claim_year and claim.schema and claim.schema.time_period:
        m = re.search(r"(\d{4})", claim.schema.time_period)
        if m:
            claim_year = m.group(1)
    # [F1] schema.time_period에서 연-월 추출도 시도
    if not claim_year_month and claim.schema and claim.schema.time_period:
        ym = re.search(r"(\d{4})[-/]?(\d{2})", claim.schema.time_period)
        if ym:
            claim_year_month = ym.group(1) + ym.group(2)

    # [v6.14 C2] 증가율/차이 자동 계산 분기
    # claim이 증가율(%) 또는 변화량(차이) schema이고 prev_value가 있으면,
    # KOSIS에서 *절대값* row를 가져와서 *우리가 직접 계산* → claim과 비교.
    # 이렇게 안 하면 catalog가 절대값 표만 매칭해서 *단위 type 불일치*로 차단됨.
    prev_value = getattr(claim.schema, "prev_value", None) if claim.schema else None
    if prev_value is not None and prev_value != 0:
        indicator = (claim.schema.indicator or "") if claim.schema else ""
        is_ratio_schema = claim_unit and ("%" in claim_unit or "퍼센트" in claim_unit
                                          or "율" in claim_unit or "비율" in claim_unit)
        is_diff_schema = ("차이" in indicator or "증감" in indicator
                          or "변화량" in indicator)
        if is_ratio_schema or is_diff_schema:
            calc_result = _verify_growth_or_diff(
                claim, evidence, claim_year, claim_year_month,
                prev_value, is_ratio_schema, config,
            )
            if calc_result is not None:
                return calc_result
            # 계산 실패 시 (KOSIS에서 현재값 못 찾음 등) 일반 분기로 fallthrough

    # ── 전체 행 탐색 (factcheck_test.py v7 핵심) ──────────────────────────
    raw = evidence.raw_response if isinstance(evidence.raw_response, dict) else {}
    rows = raw.get("row", [])
    if isinstance(rows, list) and rows:
        kosis_values = _extract_numeric_values(rows)
        if kosis_values:
            best_match, best_error = _find_best_match(
                claimed, claim_unit, claim_year, kosis_values,
                claim_year_month=claim_year_month,
            )

            if best_match is None:
                return VerificationResult(
                    claim_id=claim.claim_id, verdict=VerdictType.UNVERIFIABLE,
                    confidence=0.3, evidence=evidence)

            # [v6.14 F2] best_match로 evidence 덮어쓰기 — 프론트 표시 동기화
            # 이전: 프론트엔 *대표 cell (first cell)*만 표시 → verifier가 비교한 row와
            #       다른 row가 보여서 사용자 혼란.
            # 이제: 실제 매칭된 row의 (value, unit, period)를 evidence에 반영.
            evidence = evidence.model_copy(update={
                "official_value": best_match.get("value"),
                "unit": best_match.get("unit") or evidence.unit,
                "time_period": best_match.get("period") or evidence.time_period,
            })
            logger.info(
                f"[verifier] evidence 동기화 (F2): official_value={evidence.official_value} "
                f"unit={evidence.unit!r} time_period={evidence.time_period!r}"
            )

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
    # [v6.14 H fix] 분모 1 버그 수정 — 상위 _find_best_match와 동일 공식
    denom = max(abs(official), abs(claimed), 1e-9)
    diff_pct = abs(claimed - official) / denom * 100

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

# ── [v6.14 C2] 증가율/차이 자동 계산 ──────────────────────────────────────

def _verify_growth_or_diff(
    claim: Claim,
    evidence: Evidence,
    claim_year: str | None,
    claim_year_month: str | None,
    prev_value: float,
    is_ratio: bool,
    config: dict,
) -> VerificationResult | None:
    """
    증가율/차이 schema의 자동 계산 검증.

    프로세스:
      1. KOSIS raw_response에서 *현재 시점의 절대값* row 찾기 (unit 검사 우회, 시점 우선)
      2. claim의 *prev_value*와 함께 계산:
         - 증가율(%): (current - prev) / prev * 100
         - 차이(절대): current - prev
      3. claim의 value와 비교 → verdict

    KOSIS 현재 시점 row를 못 찾으면 None 반환 (호출자가 일반 분기로 fallthrough).

    [구체적 케이스]
      - "출생아 수 6.7% 증가, prev=19059" → KOSIS 2025-04 출생아 수 row 찾기
        → 만약 20171이면 (20171-19059)/19059*100 = 5.83%
        → claim 6.7% vs 5.83% → 비교
      - "합계출산율 차이 0.04, prev=0.72" → KOSIS 2025-04 합계출산율 row 찾기
        → 만약 0.76이면 0.76-0.72 = 0.04 → claim 0.04 vs 0.04 → MATCH
    """
    claimed = claim.schema.value
    raw = evidence.raw_response if isinstance(evidence.raw_response, dict) else {}
    rows = raw.get("row", [])
    if not isinstance(rows, list) or not rows:
        return None

    kosis_values = _extract_numeric_values(rows)
    if not kosis_values:
        return None

    # 현재 시점 row 찾기 — unit 검사 우회 (절대값 row 받아들임)
    # 시점 tier 우선
    tier1, tier2, tier3 = [], [], []
    for kv in kosis_values:
        kv_period = kv.get("period") or ""
        normalized = normalize_value(kv["value"], kv["unit"])
        if normalized == 0:
            continue
        # 연도 ±2년만 필터 (unit 검사는 *우회*)
        if claim_year and kv_period:
            try:
                if abs(int(claim_year) - int(kv_period[:4])) > 2:
                    continue
            except (ValueError, TypeError):
                pass
        kv_norm = {**kv, "normalized": normalized}
        if claim_year_month and kv_period.startswith(claim_year_month):
            tier1.append(kv_norm)
        elif claim_year and kv_period.startswith(claim_year):
            tier2.append(kv_norm)
        else:
            tier3.append(kv_norm)

    pool = tier1 or tier2 or tier3
    if not pool:
        logger.info(f"[verifier C2] 증가율 계산: KOSIS에서 현재 시점 row 못 찾음. fallthrough.")
        return None

    # tier 안에서 *prev_value 자릿수와 비슷한 row* 선택 (안전)
    # 예: prev_value=19059 (5자리) → 비슷한 자릿수의 row picking
    # 예: prev_value=0.72 (소수) → 소수 row picking
    def _scale_match(kv):
        v = abs(kv["normalized"])
        p = abs(prev_value)
        if v == 0 or p == 0:
            return float("inf")
        return abs(v / p - 1) if v >= p else abs(p / v - 1)

    current_row = min(pool, key=_scale_match)
    current_value = current_row["normalized"]

    # 계산
    if is_ratio:
        calculated = (current_value - prev_value) / prev_value * 100
        calc_desc = f"증가율 ({current_value} - {prev_value}) / {prev_value} * 100 = {calculated:.2f}%"
    else:
        calculated = current_value - prev_value
        calc_desc = f"차이 {current_value} - {prev_value} = {calculated:.4f}"

    # [H fix] 분모 1 버그 회피한 공식
    denom = max(abs(calculated), abs(claimed), 1e-9)
    error_rate = abs(calculated - claimed) / denom

    logger.info(
        f"[verifier C2] {calc_desc} | claim={claimed} | error_rate={error_rate*100:.2f}% | "
        f"current_row: period={current_row.get('period')!r} value={current_row['value']} "
        f"unit={current_row.get('unit')!r}"
    )

    # evidence 동기화 (F2 패턴) — current_row 정보로 덮어쓰기
    evidence = evidence.model_copy(update={
        "official_value": current_row.get("value"),
        "unit": current_row.get("unit") or evidence.unit,
        "time_period": current_row.get("period") or evidence.time_period,
    })

    # _verdict_from_error에 best_match 정보 전달
    best_match_info = {
        **current_row,
        "error_rate": error_rate,
        "calculated_from_prev": calculated,
        "prev_value": prev_value,
    }
    return _verdict_from_error(claim, evidence, error_rate, best_match_info, config)
