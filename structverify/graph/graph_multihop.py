"""
graph/graph_multihop.py — Multi-hop GraphRAG 파생 주장 검증 (Step 7.5)

[김예슬 - 2026-05-08 / v1]
KOSIS에서 직접 검증 불가능한 "파생 주장"을 Graph 2-hop 탐색으로 검증.

[문제]
  "쉬었음 청년이 20년 새 2.6배 늘었다" 같은 비율/배수 주장은
  KOSIS에 "2.6배"라는 수치가 직접 존재하지 않음 → 항상 unverifiable.

[해결 — Multi-hop GraphRAG]
  COMPARE 엣지로 같은 indicator를 공유하는 Claim들을 연결한 뒤,
  파생 주장(C3: "2.6배")을 그 원천 주장(C1, C2)들의 검증된 수치로 재계산.

  탐색 흐름 (2-hop):
    C3("2.6배") -[BELONGS_TO]-> MetricNode("쉬었음인구")
                                    ^
                                    | [BELONGS_TO]
                            C1(21만7천, 2024), C2(8만4천, 2004)
    → C1, C2가 각각 KOSIS로 MATCH 판정됨
    → 217000 / 84000 = 2.58 ≈ 2.6 → C3도 MATCH

[설계 원칙]
  - LLM 미사용 — 비율/배수/차이 계산은 deterministic
  - C1, C2가 먼저 검증(MATCH/MISMATCH)되어 있어야 함 → Step 8 이후 실행
  - 원천 주장이 검증 안 됐으면 multi-hop도 unverifiable

[참고] GraphRAG (Edge et al., 2024) — 엔티티 그래프 기반 멀티홉 추론
"""
from __future__ import annotations

import re

from structverify.core.schemas import (
    Claim, ClaimType, GraphEdge, VerdictType, VerificationResult,
)
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


# ── 파생 주장 판별 ──────────────────────────────────────────────────────────

# 배수/비율 표현 패턴 — "2.6배", "두 배", "3분의 1" 등
_RATIO_PATTERNS = [
    re.compile(r"(\d+\.?\d*)\s*배"),           # 2.6배
    re.compile(r"(\d+\.?\d*)\s*퍼센트|%"),     # 30%
]
# 차이 표현 — "8만→22만", "8만에서 22만으로"
_DIFF_PATTERNS = [
    re.compile(r"(\d[\d,]*)\s*[만천]?\s*[→~]\s*(\d[\d,]*)"),
]


def is_derived_claim(claim: Claim) -> bool:
    """
    파생 주장(비율/배수/차이)인지 판별.

    파생 주장:
      - canonical_type이 SCALE / COMPARISON
      - claim_text에 "배", "배로", "분의" 등 비율 표현
      - schema.value가 비율값으로 보임 (0 < value < 100, unit이 "배"/"%")
    """
    text = claim.claim_text or ""

    # 1) canonical_type 기반
    if claim.canonical_type in (ClaimType.SCALE, ClaimType.COMPARISON):
        return True

    # 2) 텍스트 패턴 — "N배"
    if re.search(r"\d+\.?\d*\s*배", text):
        return True

    # 3) unit이 "배"
    if claim.schema and claim.schema.unit:
        if "배" in claim.schema.unit:
            return True

    return False


def extract_ratio_value(claim: Claim) -> float | None:
    """
    파생 주장에서 비율/배수 값을 추출.
    "2.6배 늘었다" → 2.6
    """
    if claim.schema and claim.schema.value is not None:
        v = claim.schema.value
        if 0 < v < 1000:   # 비율로 보이는 범위
            return v

    text = claim.claim_text or ""
    m = re.search(r"(\d+\.?\d*)\s*배", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


# ── COMPARE 엣지 2-hop 탐색 ─────────────────────────────────────────────────

def find_compare_neighbors(
    target_claim: Claim,
    all_claims: list[Claim],
    edges: list[GraphEdge],
) -> list[Claim]:
    """
    target_claim과 COMPARE 엣지로 연결된 Claim들을 반환.

    Graph 2-hop:
      target -[BELONGS_TO]-> MetricNode <-[BELONGS_TO]- neighbor
      (graph_builder가 이미 COMPARE 엣지로 표현해둠)

    Args:
        target_claim: 파생 주장
        all_claims:   전체 claim 목록
        edges:        graph_builder가 만든 엣지 목록

    Returns:
        같은 indicator를 공유하는 다른 Claim들
    """
    target_nid = f"claim:{target_claim.claim_id.hex[:8]}"

    # COMPARE 엣지에서 target과 연결된 노드 id 수집
    neighbor_nids: set[str] = set()
    for edge in edges:
        etype = edge.edge_type
        etype_val = etype.value if hasattr(etype, "value") else etype
        if etype_val != "compare":
            continue
        if edge.from_node == target_nid:
            neighbor_nids.add(edge.to_node)
        elif edge.to_node == target_nid:
            neighbor_nids.add(edge.from_node)

    if not neighbor_nids:
        return []

    # node_id → Claim 매핑
    neighbors = []
    for claim in all_claims:
        nid = f"claim:{claim.claim_id.hex[:8]}"
        if nid in neighbor_nids:
            neighbors.append(claim)

    return neighbors


# ── Multi-hop 검증 ──────────────────────────────────────────────────────────

def verify_via_multihop(
    target_claim: Claim,
    all_claims: list[Claim],
    edges: list[GraphEdge],
    results_by_claim: dict[str, VerificationResult],
    config: dict | None = None,
) -> VerificationResult | None:
    """
    파생 주장을 COMPARE 이웃들의 검증된 수치로 재검증.

    [핵심 로직]
      C3: "2.6배 늘었다" (파생 주장)
        → COMPARE 이웃: C1(217000, 2024), C2(84000, 2004)
        → C1, C2가 모두 검증됨 (official_value 존재)
        → 큰값 / 작은값 = 217000 / 84000 = 2.58
        → claimed 2.6 vs computed 2.58 → 오차 0.8% → MATCH

    Args:
        target_claim:      검증할 파생 주장
        all_claims:        전체 claim
        edges:             graph 엣지
        results_by_claim:  {claim_id_hex: VerificationResult} — Step 8 결과
        config:            설정 (tolerance 등)

    Returns:
        재검증된 VerificationResult, 또는 multi-hop 불가 시 None
    """
    config = config or {}
    tolerance = config.get("verification", {}).get("multihop_tolerance_pct", 10.0)

    # 1) 파생 주장인지 확인
    if not is_derived_claim(target_claim):
        return None

    claimed_ratio = extract_ratio_value(target_claim)
    if claimed_ratio is None:
        logger.debug(f"파생 주장이지만 비율값 추출 실패: {target_claim.claim_text[:40]}")
        return None

    # 2) COMPARE 이웃 탐색 (2-hop)
    neighbors = find_compare_neighbors(target_claim, all_claims, edges)
    if len(neighbors) < 2:
        logger.debug(
            f"Multi-hop 불가 — COMPARE 이웃 부족 "
            f"({len(neighbors)}개): {target_claim.claim_text[:40]}"
        )
        return None

    # 3) 이웃들 중 검증된(official_value 있는) 수치 수집
    verified_values: list[tuple[float, str]] = []   # (official_value, time_period)
    for nb in neighbors:
        nb_hex = nb.claim_id.hex
        nb_result = results_by_claim.get(nb_hex)
        if not nb_result or not nb_result.evidence:
            continue
        official = nb_result.evidence.official_value
        if official is None or official == 0:
            continue
        # 원천 주장이 MATCH/MISMATCH(=수치 확인됨)인 것만 사용
        if nb_result.verdict == VerdictType.UNVERIFIABLE:
            continue
        period = ""
        if nb.schema and nb.schema.time_period:
            period = nb.schema.time_period
        verified_values.append((official, period))

    if len(verified_values) < 2:
        logger.debug(
            f"Multi-hop 불가 — 검증된 이웃 수치 부족 "
            f"({len(verified_values)}개): {target_claim.claim_text[:40]}"
        )
        return None

    # 4) 비율 계산 — 큰 값 / 작은 값
    values_only = sorted([v for v, _ in verified_values], reverse=True)
    largest = values_only[0]
    smallest = values_only[-1]

    if smallest == 0:
        return None

    computed_ratio = largest / smallest

    # 5) 판정 — claimed vs computed
    diff_pct = abs(claimed_ratio - computed_ratio) / max(computed_ratio, 1e-9) * 100

    if diff_pct <= tolerance:
        verdict = VerdictType.MATCH
        conf = min(0.9, 1.0 - diff_pct / 100)
    elif diff_pct <= 30:
        verdict = VerdictType.UNVERIFIABLE
        conf = 0.4
    else:
        verdict = VerdictType.MISMATCH
        conf = min(0.85, diff_pct / 100)

    logger.info(
        f"[Multi-hop] {target_claim.claim_text[:40]} → "
        f"claimed={claimed_ratio:.2f}배 vs computed={computed_ratio:.2f}배 "
        f"({largest:.0f}/{smallest:.0f}) → {verdict.value} (오차 {diff_pct:.1f}%)"
    )

    # 원천 주장 evidence를 멀티홉 결과에 첨부 (가장 큰 값 쪽)
    source_evidence = None
    for nb in neighbors:
        nb_result = results_by_claim.get(nb.claim_id.hex)
        if nb_result and nb_result.evidence:
            if nb_result.evidence.official_value == largest:
                source_evidence = nb_result.evidence
                break

    result = VerificationResult(
        claim_id=target_claim.claim_id,
        verdict=verdict,
        confidence=conf,
        evidence=source_evidence,
    )
    # multi-hop 사용 표시 (explainer가 참고)
    result.multihop_used = True
    result.multihop_detail = {
        "claimed_ratio":  claimed_ratio,
        "computed_ratio": round(computed_ratio, 3),
        "largest_value":  largest,
        "smallest_value": smallest,
        "neighbor_count": len(verified_values),
    }
    return result


def apply_multihop_verification(
    claims: list[Claim],
    results: list[VerificationResult],
    edges: list[GraphEdge],
    config: dict | None = None,
) -> list[VerificationResult]:
    """
    Step 8 이후 호출 — UNVERIFIABLE인 파생 주장들을 multi-hop으로 재검증.

    runtime_agent에서:
      results = [verify_claim(c, ev) for c in claims]   # Step 8
      results = apply_multihop_verification(claims, results, edges)  # Step 8.5

    Args:
        claims:  전체 claim
        results: Step 8 검증 결과 (claims와 같은 순서)
        edges:   graph 엣지 (COMPARE 포함)
        config:  설정

    Returns:
        multi-hop 재검증이 반영된 results (UNVERIFIABLE → MATCH/MISMATCH 가능)
    """
    # claim_id_hex → result 매핑
    results_by_claim = {
        str(r.claim_id.hex if hasattr(r.claim_id, "hex") else r.claim_id): r
        for r in results
    }
    # claim 순서 유지용
    result_by_idx = {i: r for i, r in enumerate(results)}

    multihop_count = 0
    for idx, claim in enumerate(claims):
        current = result_by_idx[idx]

        # 이미 MATCH/MISMATCH면 multi-hop 불필요
        if current.verdict != VerdictType.UNVERIFIABLE:
            continue

        # 파생 주장만 시도
        if not is_derived_claim(claim):
            continue

        mh_result = verify_via_multihop(
            claim, claims, edges, results_by_claim, config
        )
        if mh_result is not None:
            result_by_idx[idx] = mh_result
            results_by_claim[claim.claim_id.hex] = mh_result
            multihop_count += 1

    if multihop_count:
        logger.info(f"[Multi-hop] {multihop_count}건 재검증 완료")

    return [result_by_idx[i] for i in range(len(results))]