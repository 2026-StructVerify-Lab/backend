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
    Claim, Evidence, VerificationResult, VerdictType, MismatchType, ValueRole)
from structverify.verification.combiners import (
    combine, COMBINER_DIRECT, COMBINER_DELTA, COMBINER_RATIO_PCT,
)
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


# [v6.2] row 컨텍스트 토큰 추출용 컬럼 — KOSIS 표준 분류 컬럼들
_ROW_CONTEXT_COLUMNS = ("C1_NM", "C2_NM", "C3_NM", "C4_NM", "ITM_NM")

# 광역시도 레퍼런스 — claim이 "전국"이면 이 중 하나가 row.C1_NM에 있을 때 부적합
_SIDO_TOKENS = (
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
)
_GLOBAL_POPULATION_TOKENS = ("전국", "대한민국", "한국", "전체", "전 국민", "국내")


def _row_population_compatible(raw_row: dict, claim_population: str | None) -> bool:
    """
    [v6.2] row의 지역 컨텍스트와 claim의 population이 호환되는지.

    명백한 충돌만 잡음:
      - claim="전국/한국/전체" + row.C1_NM이 특정 시도 → 부적합
      - 그 외 모든 케이스(정보 부족 등)는 통과

    이건 도메인 룰이 아니라 일반적 컨텍스트 매칭. KOSIS 표는 시도별 row가
    섞여 있는 경우가 많아 best 매칭이 우연히 같은 값을 찾는 가짜 match를 차단.
    """
    if not claim_population:
        return True
    region = (raw_row.get("C1_NM") or "").strip()
    if not region:
        return True

    pop = claim_population.lower()
    rg = region.lower()

    is_claim_global = any(tok in pop for tok in _GLOBAL_POPULATION_TOKENS)
    is_row_sido = any(tok in rg for tok in _SIDO_TOKENS)

    if is_claim_global and is_row_sido:
        return False  # 전국 vs 특정 시도 → 차단

    # 역방향: claim에 특정 시도가 있는데 row가 다른 시도 → 부적합
    claim_sidos = [tok for tok in _SIDO_TOKENS if tok in pop]
    if claim_sidos and is_row_sido:
        if not any(tok in rg for tok in claim_sidos):
            return False  # claim="서울" + row="부산" → 차단

    return True


def _find_best_match(
    claimed: float,
    claim_unit: str,
    claim_year: str | None,
    kosis_values: list[dict],
    claim_population: str | None = None,
) -> tuple[dict | None, float]:
    """
    KOSIS 전체 행에서 claim과 가장 가까운 값 탐색.

    [v6.2] claim_population을 받아 row의 지역 컨텍스트 호환성 체크 추가.
           "전국" claim에 시도별 row가 매칭되는 가짜 match 차단.

    Returns:
        (best_match_dict, best_error_rate)
    """
    best_match = None
    best_error = float("inf")
    skipped_ctx = 0

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

        # [v6.2] row 컨텍스트 호환성 체크
        if not _row_population_compatible(kv.get("raw", {}), claim_population):
            skipped_ctx += 1
            continue

        error_rate = abs(normalized - claimed) / max(abs(claimed), 1)
        if error_rate < best_error:
            best_error = error_rate
            best_match = {**kv, "normalized": normalized, "error_rate": error_rate}

    if skipped_ctx:
        logger.debug(f"_find_best_match: 컨텍스트 충돌로 {skipped_ctx}개 row 제외")

    return best_match, best_error


# ── 메인 검증 함수 ─────────────────────────────────────────────────────────────

def verify_claim(claim: Claim, evidence: Evidence | None,
                 config: dict | None = None,
                 graph: "ClaimGraph | None" = None,
                 evidences: list[Evidence] | None = None) -> VerificationResult:
    """
    공식 통계와 기사 수치를 비교하여 판정 (LLM 미사용).

    [v3] factcheck_test.py v7 로직 전면 반영
    [v6 멀티홉] graph가 있으면 claim의 시점을 그래프에서 resolved된 절대 시점으로
                보정하여 KOSIS row 매칭에 사용.
    [v6.3] evidences 인자(list) 지원 — delta/ratio 검증을 위한 multi-evidence.
            value_role이 measurement면 evidences[0] 또는 evidence를 사용 (호환).
            delta/ratio이면 combiner로 결합.
    """
    config = config or {}

    # evidences 정규화: list 우선, 없으면 단일 evidence를 list로 wrap
    if evidences is None:
        evidences = [evidence] if evidence is not None else []
    elif evidence is None and evidences:
        # 호환성: primary/endpoint_a evidence를 단일 evidence 슬롯에도 넣기
        evidence = next(
            (e for e in evidences if e.requirement_role in ("primary", "endpoint_a")),
            evidences[0],
        )

    # ── [v6.3] value_role/combiner 기반 분기 ────────────────────────────
    schema = claim.schema
    value_role = schema.value_role if schema else ValueRole.MEASUREMENT
    plan = schema.evidence_plan if schema else None
    combiner_name = (plan.combiner if plan else None) or COMBINER_DIRECT

    # threshold/rank/none 등 검증 부적합 케이스
    if value_role in (ValueRole.THRESHOLD, ValueRole.RANK, ValueRole.NONE):
        logger.info(
            f"value_role={value_role.value} → KOSIS 직접 비교 제외 "
            f"(claim={claim.sent_id})"
        )
        return VerificationResult(
            claim_id=claim.claim_id,
            verdict=VerdictType.UNVERIFIABLE,
            confidence=0.5,
            evidence=evidence,
            supplementary_evidences=list(evidences),
            combiner_used=combiner_name,
        )

    # delta/ratio: combiner 분기
    if combiner_name in (COMBINER_DELTA, COMBINER_RATIO_PCT):
        return _verify_with_combiner(
            claim, evidences, combiner_name, evidence,
        )

    # 그 외 (measurement / direct): 기존 단일 evidence 비교 로직
    return _verify_direct(claim, evidence, evidences, combiner_name, graph=graph, config=config)


def _verify_with_combiner(
    claim: Claim,
    evidences: list[Evidence],
    combiner_name: str,
    primary_evidence: Evidence | None,
) -> VerificationResult:
    """
    [v6.3] delta/ratio 검증.

    1. evidences를 combiner로 결합 → computed_value
    2. computed_value vs claim.schema.value 비교
    3. 오차에 따라 verdict 결정 (단일 비교와 동일한 임계)
    """
    schema = claim.schema
    if schema is None or schema.value is None:
        return VerificationResult(
            claim_id=claim.claim_id, verdict=VerdictType.UNVERIFIABLE,
            confidence=0.2, evidence=primary_evidence,
            supplementary_evidences=list(evidences),
            combiner_used=combiner_name,
        )

    # endpoint 부족 → 검증 불가
    if len(evidences) < 2:
        logger.info(
            f"combiner={combiner_name}: evidence {len(evidences)}개 — 2개 필요 "
            f"→ unverifiable (claim={claim.sent_id})"
        )
        return VerificationResult(
            claim_id=claim.claim_id, verdict=VerdictType.UNVERIFIABLE,
            confidence=0.3, evidence=primary_evidence,
            supplementary_evidences=list(evidences),
            combiner_used=combiner_name,
        )

    computed, formula = combine(combiner_name, evidences)
    if computed is None:
        logger.warning(f"combiner={combiner_name} 결합 실패: {formula}")
        return VerificationResult(
            claim_id=claim.claim_id, verdict=VerdictType.UNVERIFIABLE,
            confidence=0.3, evidence=primary_evidence,
            supplementary_evidences=list(evidences),
            combiner_used=combiner_name,
        )

    claimed = float(schema.value)

    # 오차율 (절대값 + 0 division 회피)
    denom = max(abs(claimed), abs(computed), 1e-6)
    error_rate = abs(computed - claimed) / denom

    logger.info(
        f"[combiner] {combiner_name}: claimed={claimed} computed={computed:+.4g} "
        f"오차={error_rate*100:.1f}% formula={formula}"
    )

    # 임계 (단일 비교와 동일)
    if error_rate <= 0.10:
        verdict = VerdictType.MATCH
        confidence = 0.95 - error_rate
    elif error_rate <= 0.30:
        verdict = VerdictType.UNVERIFIABLE
        confidence = 0.5
    elif error_rate <= 0.90:
        verdict = VerdictType.MISMATCH
        confidence = min(0.95, 0.5 + error_rate / 2)
    else:
        verdict = VerdictType.UNVERIFIABLE
        confidence = 0.3

    return VerificationResult(
        claim_id=claim.claim_id,
        verdict=verdict,
        confidence=confidence,
        evidence=primary_evidence,
        supplementary_evidences=list(evidences),
        computed_value=computed,
        combiner_used=combiner_name,
        mismatch_type=MismatchType.VALUE if verdict == VerdictType.MISMATCH else None,
    )


def _verify_direct(
    claim: Claim,
    evidence: Evidence | None,
    evidences: list[Evidence],
    combiner_name: str,
    graph: "ClaimGraph | None" = None,
    config: dict | None = None,
) -> VerificationResult:
    """기존 단일 evidence 비교 로직 (measurement/direct)."""
    config = config or {}

    # evidence 없음
    if evidence is None or evidence.official_value is None:
        return VerificationResult(
            claim_id=claim.claim_id, verdict=VerdictType.UNVERIFIABLE,
            confidence=0.3, evidence=evidence,
            supplementary_evidences=list(evidences),
            combiner_used=combiner_name,
        )

    # claim schema 없음
    claimed = claim.schema.value if claim.schema else None
    if claimed is None:
        return VerificationResult(
            claim_id=claim.claim_id, verdict=VerdictType.UNVERIFIABLE,
            confidence=0.2, evidence=evidence,
            supplementary_evidences=list(evidences),
            combiner_used=combiner_name,
        )

    # value=0.0 → 수치 미추출
    if claimed == 0.0:
        return VerificationResult(
            claim_id=claim.claim_id, verdict=VerdictType.UNVERIFIABLE,
            confidence=0.2, evidence=evidence,
            supplementary_evidences=list(evidences),
            combiner_used=combiner_name,
        )

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
            # [v6.2] claim_population 전달 → 전국 vs 시도 충돌 row 제외
            claim_population = (claim.schema.population if claim.schema else None)
            best_match, best_error = _find_best_match(
                claimed, claim_unit, claim_year, kosis_values,
                claim_population=claim_population,
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