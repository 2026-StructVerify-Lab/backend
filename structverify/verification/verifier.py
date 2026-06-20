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
# [2026-05-14 | 이수민] memory/v1: working memory 도메인 가드 추가
#   - verify_claim() 시그니처에 memory: DocumentWorkingMemory 인자 추가
#   - evidence.category_path와 memory.domain 불일치 시 UNVERIFIABLE(DOMAIN_MISMATCH) 반환
#   - 거절된 stat_id는 memory.rejected_stat_ids에 기록 (false match 방지)
from __future__ import annotations

import re

from typing import TYPE_CHECKING

from structverify.core.schemas import (
    Claim, Evidence, VerificationResult, VerdictType, MismatchType)
from structverify.utils.logger import get_logger
# [리팩] fallback 오차 구간 판정 → verdict_thresholds.py
from .verdict_thresholds import verdict_from_error as _verdict_from_error_impl
# [리팩] 단위·row 매칭 → units.py / row_match.py
from .units import is_same_unit_type, normalize_value
from .row_match import (
    extract_numeric_values as _extract_numeric_values,
    find_best_match as _find_best_match,
)
# [리팩] 증가율/차이 계산 → growth_diff.py
from .growth_diff import verify_growth_or_diff as _verify_growth_or_diff_impl

if TYPE_CHECKING:
    from structverify.memory.working_memory import DocumentWorkingMemory
    from structverify.graph.claim_graph import ClaimGraph

logger = get_logger(__name__)


# ── 메인 검증 함수 ─────────────────────────────────────────────────────────────

def verify_claim(claim: Claim, evidence: Evidence | None,
                 config: dict | None = None,
                 graph: "ClaimGraph | None" = None,
                 memory: "DocumentWorkingMemory | None" = None) -> VerificationResult:
    """
    공식 통계와 기사 수치를 비교하여 판정 (LLM 미사용).

    [v3] factcheck_test.py v7 로직 전면 반영
    [v6 멀티홉] graph가 있으면 claim의 시점을 그래프에서 resolved된 절대 시점으로
                보정하여 KOSIS row 매칭에 사용. claim.schema.time_period가
                "작년" 같은 상대 표현이어도 그래프 traverse로 2023이 나옴.
    [v7 이수민 2026-05-14] memory 도메인 가드:
                memory가 있고 evidence.category_path가 문서 도메인과 어긋나면
                DOMAIN_MISMATCH로 UNVERIFIABLE 반환 (false match 방지).
    """
    config = config or {}

    # evidence 없음
    if evidence is None or evidence.official_value is None:
        return VerificationResult(
            claim_id=claim.claim_id, verdict=VerdictType.UNVERIFIABLE,
            confidence=0.3, evidence=evidence)

    # claim schema 없음 / value 미추출
    # [v6.16] value가 없으면 검증 자체를 안 한 것 → 엉뚱한 KOSIS 출처를
    #   화면에 남기지 않도록 evidence=None (예: value=null인데 '종합부동산세
    #   196059 백만원'이 출처로 표시되던 문제)
    claimed = claim.schema.value if claim.schema else None
    if claimed is None:
        return VerificationResult(
            claim_id=claim.claim_id, verdict=VerdictType.UNVERIFIABLE,
            confidence=0.2, evidence=None)

    # value=0.0 → 수치 미추출
    if claimed == 0.0:
        return VerificationResult(
            claim_id=claim.claim_id, verdict=VerdictType.UNVERIFIABLE,
            confidence=0.2, evidence=None)

    # ── [v7 이수민 2026-05-14] 도메인 가드 ─────────────────────────────
    # working memory의 doc 도메인과 evidence의 카테고리가 일치하는지 확인.
    # 어긋나면 잘못된 stat_id로 매칭된 false match 가능성 → UNVERIFIABLE.
    if memory is not None and evidence.category_path:
        if not memory.domain_matches_category(evidence.category_path):
            logger.info(
                f"[verifier 도메인 가드] reject: "
                f"doc.domain={memory.domain} ↔ evidence.category={evidence.category_path}"
            )
            memory.record_stat_id_rejected(
                evidence.stat_table_id or "unknown",
                f"domain mismatch: {memory.domain} vs {evidence.category_path}",
            )
            return VerificationResult(
                claim_id=claim.claim_id,
                verdict=VerdictType.UNVERIFIABLE,
                confidence=0.4,
                evidence=evidence,
                mismatch_type=MismatchType.DOMAIN_MISMATCH,
            )

    claim_unit = (claim.schema.unit or "") if claim.schema else ""

    # ── 연도 추출 — [v6] 그래프 우선, 그 다음 schema.time_period ────────
    claim_year = None
    claim_year_month = None  # [v6.14 F1] 동일 연-월 우선 매칭용 (예: "202504")

    # ── 연도/연-월 추출 ─────────────────────────────────────────────────
    # [v6.14 I fix] 우선순위 역전:
    #   1) claim.schema.time_period가 *구체적 시점* (YYYY 또는 YYYY-MM)이면 *우선* 사용
    #   2) schema가 비어있거나 *상대 표현* ("올해", "작년" 등)이면 graph traversal로 fallback
    #
    # 이전 (v6.14 H 이전): graph가 우선이라 한 문장에 두 시점이 있을 때
    #   ("올 4월 ... 지난해 같은 달") 그래프가 *지난해 같은 달*을 picking하면
    #   verifier가 *2024*로 검색해서 시점 부정확 매칭 발생.
    # 수정: schema_inductor가 *문맥 보고 추출한* time_period가 보통 정확하므로 우선.
    #
    # [F1] 연도와 연-월 모두 추출 시도.

    schema_tp = (claim.schema.time_period if claim.schema and claim.schema.time_period else "")

    # 1) schema에서 *구체적 시점* 추출 시도
    if schema_tp:
        m = re.search(r"(\d{4})", schema_tp)
        if m:
            claim_year = m.group(1)
        ym = re.search(r"(\d{4})[-/]?(\d{2})", schema_tp)
        if ym:
            claim_year_month = ym.group(1) + ym.group(2)
        if claim_year:
            logger.info(f"[verifier] 시점 해소: schema.time_period={schema_tp!r} → year={claim_year}, ym={claim_year_month}")

    # 2) schema에서 시점 못 뽑으면 (또는 schema가 *상대 표현*만 있으면) → graph fallback
    if not claim_year and graph is not None:
        resolved = graph.resolve_time_for_claim(claim)
        if resolved:
            m = re.search(r"(\d{4})", resolved)
            if m:
                claim_year = m.group(1)
                logger.info(f"[verifier] 시점 해소 (fallback): 그래프에서 resolved year={claim_year} (from {resolved})")
            ym = re.search(r"(\d{4})[-/]?(\d{2})", resolved)
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
            calc_result = _verify_growth_or_diff_impl(
                claim, evidence, claim_year, claim_year_month,
                prev_value, is_ratio_schema, config,
                classify_mismatch=_classify_mismatch,
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
    """[리팩] 오차율 → 판정 — verdict_thresholds.verdict_from_error 위임."""
    return _verdict_from_error_impl(
        claim, evidence, error_rate, best_match, config, _classify_mismatch,
    )


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

