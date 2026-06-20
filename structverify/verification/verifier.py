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

from typing import TYPE_CHECKING

from structverify.core.schemas import Claim, Evidence, VerificationResult
# [리팩] Evidence → NormalizedInput
from .adapters import from_evidence
# [리팩] 판정 메인 진입
from .decide_verdict import decide_verdict

if TYPE_CHECKING:
    from structverify.memory.working_memory import DocumentWorkingMemory
    from structverify.graph.claim_graph import ClaimGraph


def verify_claim(
    claim: Claim,
    evidence: Evidence | None,
    config: dict | None = None,
    graph: "ClaimGraph | None" = None,
    memory: "DocumentWorkingMemory | None" = None,
) -> VerificationResult:
    """
    공식 통계와 기사 수치를 비교하여 판정 (LLM 미사용).

    [v3] factcheck_test.py v7 로직 전면 반영
    [v6 멀티홉] graph가 있으면 claim의 시점을 그래프에서 resolved된 절대 시점으로
                보정하여 KOSIS row 매칭에 사용. claim.schema.time_period가
                "작년" 같은 상대 표현이어도 그래프 traverse로 2023이 나옴.
    [v7 이수민 2026-05-14] memory 도메인 가드:
                memory가 있고 evidence.category_path가 문서 도메인과 어긋나면
                DOMAIN_MISMATCH로 UNVERIFIABLE 반환 (false match 방지).
  [리팩] 판정 본문 → decide_verdict(profile="fallback") 위임.
    """
    config = config or {}
    normalized, early = from_evidence(
        claim, evidence, graph=graph, memory=memory,
    )
    if early is not None:
        return early
    return decide_verdict(claim, normalized, config, profile="fallback")
