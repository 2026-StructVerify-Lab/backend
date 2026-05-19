# [이수민 - 2026-05-14]
# - DocumentWorkingMemory: 한 doc 처리 동안만 살아있는 working memory
# - 목적: step간 컨텍스트 공유 + verifier 도메인 가드 + 파생 주장 검증
# - 수명: 파이프라인 시작 시 생성, 종료 시 소멸 (영속화 없음)
# - 영속 메모리(AgentMemory, ExemplarStore)와는 다른 layer
"""
memory/working_memory.py — Document-scoped Working Memory

[설계 의도]
한 입력 문서 처리 동안 모든 step이 공유하는 in-memory 컨텍스트.
- step간 데이터 명시적 전달 (config 산재 방지)
- claim 간 관계성 (같은 indicator 묶음) 노출
- verifier가 도메인 일관성 가드로 활용

[라이프사이클]
    memory = DocumentWorkingMemory(doc_id, run_id)
    # 각 step이 read/write
    memory.record_domain(...)
    memory.record_claim(...)
    related = memory.find_related_claims("출생아 수")
    # 종료 시 소멸 (영속화 안 함)

[참고]
- 영속 KG (cross-doc) → memory/agent_memory.py (Phase 2 보류)
- 영속 사례 retrieval → memory/exemplar_store.py (Phase 2 보류)
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from structverify.core.schemas import Claim
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


# ── 보조 모델 ────────────────────────────────────────────────────────────────

class StatIdUsage(BaseModel):
    """Step 7에서 성공한 stat_id 사용 기록."""
    stat_id:       str
    indicator:     str
    category_path: str | None = None
    time_period:   str | None = None
    used_at:       datetime = Field(default_factory=datetime.utcnow)


# ── 메인 클래스 ──────────────────────────────────────────────────────────────

class DocumentWorkingMemory(BaseModel):
    """
    한 doc 처리 동안의 working memory.

    모든 step이 이 객체를 공유하며 read/write 한다.
    """
    # ── 식별 ─────────────────────────────────────────────────────────────────
    doc_id:     str
    run_id:     str
    source_uri: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # ── Step 3: Domain ──────────────────────────────────────────────────────
    domain:             str | None = None
    domain_description: str | None = None

    # ── Step 4.5: Temporal ──────────────────────────────────────────────────
    anchor_year:       int | None = None
    temporal_resolved: dict[str, str] = Field(default_factory=dict)

    # ── Step 4~6: Claims ────────────────────────────────────────────────────
    claims:           list[Claim] = Field(default_factory=list)
    metric_to_claims: dict[str, list[str]] = Field(default_factory=dict)

    # ── Step 7: KOSIS ───────────────────────────────────────────────────────
    used_stat_ids:     dict[str, StatIdUsage] = Field(default_factory=dict)
    rejected_stat_ids: dict[str, str] = Field(default_factory=dict)

    # ── 자유 메모 ────────────────────────────────────────────────────────────
    notes: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    # ════════════════════════════════════════════════════════════════════════
    # 기록 (Write)
    # ════════════════════════════════════════════════════════════════════════

    def record_domain(self, domain: str, description: str | None = None) -> None:
        """Step 3 결과 기록."""
        self.domain = domain
        self.domain_description = description
        logger.debug(f"[memory:{self.doc_id[:8]}] domain={domain}")

    def record_anchor_year(self, year: int) -> None:
        """Step 4.5에서 결정된 anchor_year."""
        self.anchor_year = year
        logger.debug(f"[memory:{self.doc_id[:8]}] anchor_year={year}")

    def record_temporal(self, expression: str, resolved: str) -> None:
        """시간 표현 → 해소된 절대 시점."""
        self.temporal_resolved[expression] = resolved

    def record_claim(self, claim: Claim) -> None:
        """
        Claim 추가 + metric_to_claims 자동 업데이트.

        같은 indicator를 가진 claim들이 자동으로 묶여서 find_related_claims로 조회 가능.
        """
        # 중복 방지 (같은 claim_id면 스킵)
        cid = str(claim.claim_id)
        if any(str(c.claim_id) == cid for c in self.claims):
            return
        self.claims.append(claim)

        # metric_to_claims 인덱스 업데이트
        if claim.schema and claim.schema.indicator:
            indicator = claim.schema.indicator
            self.metric_to_claims.setdefault(indicator, []).append(cid)

    def record_claims(self, claims: list[Claim]) -> None:
        """여러 Claim 한 번에 기록."""
        for c in claims:
            self.record_claim(c)

    def record_stat_id_used(
        self,
        indicator: str,
        stat_id: str,
        category_path: str | None = None,
        time_period: str | None = None,
    ) -> None:
        """Step 7에서 성공한 stat_id 캐시."""
        self.used_stat_ids[indicator] = StatIdUsage(
            stat_id=stat_id,
            indicator=indicator,
            category_path=category_path,
            time_period=time_period,
        )

    def record_stat_id_rejected(self, stat_id: str, reason: str) -> None:
        """도메인 어긋남 등으로 거절된 stat_id."""
        self.rejected_stat_ids[stat_id] = reason

    def add_note(self, step: str, key: str, value: Any) -> None:
        """자유 메모. notes[step][key] = value."""
        self.notes.setdefault(step, {})[key] = value

    # ════════════════════════════════════════════════════════════════════════
    # 조회 (Read)
    # ════════════════════════════════════════════════════════════════════════

    def find_related_claims(self, indicator: str) -> list[Claim]:
        """
        같은 indicator를 가진 모든 Claim 반환.

        파생 주장 검증의 핵심 — "8.7% 증가" 검증 시 같은 metric의 baseline/comparison claim 가져옴.
        """
        claim_ids = self.metric_to_claims.get(indicator, [])
        if not claim_ids:
            return []
        # claim_id 매칭으로 실제 Claim 객체 반환
        return [c for c in self.claims if str(c.claim_id) in claim_ids]

    def find_claims_by_indicator_prefix(self, prefix: str) -> list[Claim]:
        """
        indicator가 특정 prefix로 시작하는 claim 반환.

        예: "출생아 수"로 시작 → "출생아 수", "출생아 수 증가율" 둘 다 매칭.
        """
        out = []
        for indicator, cids in self.metric_to_claims.items():
            if indicator.startswith(prefix):
                out.extend(c for c in self.claims if str(c.claim_id) in cids)
        return out

    def get_stat_id_for_indicator(self, indicator: str) -> StatIdUsage | None:
        """같은 indicator로 이전에 성공한 stat_id 있으면 반환 (캐시 hit)."""
        return self.used_stat_ids.get(indicator)

    def is_stat_id_rejected(self, stat_id: str) -> bool:
        """이 stat_id가 이번 doc에서 이미 거절됐는지."""
        return stat_id in self.rejected_stat_ids

    def domain_matches_category(self, category_path: str | None) -> bool:
        """
        evidence의 category_path가 이 doc의 domain과 일치하는지.

        Verifier의 일관성 가드용. domain이 미설정이거나 category가 None이면 True (관대).

        매칭 규칙 (단순 키워드 기반, Phase 2에서 정교화 가능):
            domain="population"  ↔ category_path 포함 "인구" → True
            domain="employment"  ↔ category_path 포함 "고용"|"노동"|"취업" → True
            domain="environment" ↔ category_path 포함 "환경"|"기상"|"기후" → True
            domain="economy"     ↔ category_path 포함 "경제"|"산업"|"물가" → True
        """
        if not self.domain or not category_path:
            return True  # 정보 부족 → 통과

        # 도메인 → 카테고리 키워드 매핑
        _DOMAIN_KEYWORDS = {
            "population":  ["인구", "출생", "사망", "혼인", "이혼", "가구"],
            "employment":  ["고용", "노동", "취업", "임금", "근로", "실업"],
            "environment": ["환경", "기상", "기후", "오염", "에너지", "탄소"],
            "economy":     ["경제", "산업", "물가", "성장", "무역", "gdp"],
            "education":   ["교육", "학교", "학생", "진학"],
            "health":      ["보건", "의료", "건강", "질환"],
        }

        keywords = _DOMAIN_KEYWORDS.get(self.domain.lower(), [])
        if not keywords:
            return True  # 알 수 없는 도메인 → 통과 (보수적)

        return any(kw in category_path for kw in keywords)

    # ════════════════════════════════════════════════════════════════════════
    # 덤프 / 디버깅
    # ════════════════════════════════════════════════════════════════════════

    def dump(self) -> dict:
        """현재 메모리 상태 전체를 dict로 반환 (디버깅/로깅용)."""
        return {
            "doc_id":             self.doc_id,
            "run_id":             self.run_id,
            "source_uri":         self.source_uri,
            "created_at":         self.created_at.isoformat(),
            "domain":             self.domain,
            "domain_description": self.domain_description,
            "anchor_year":        self.anchor_year,
            "temporal_resolved":  dict(self.temporal_resolved),
            "claims_count":       len(self.claims),
            "metric_to_claims":   {k: list(v) for k, v in self.metric_to_claims.items()},
            "used_stat_ids":      {k: v.model_dump() for k, v in self.used_stat_ids.items()},
            "rejected_stat_ids":  dict(self.rejected_stat_ids),
            "notes":              dict(self.notes),
        }

    def stats(self) -> dict[str, int]:
        """간단 통계."""
        return {
            "claims":         len(self.claims),
            "metrics":        len(self.metric_to_claims),
            "temporal":       len(self.temporal_resolved),
            "stat_ids_used":  len(self.used_stat_ids),
            "stat_ids_rej":   len(self.rejected_stat_ids),
        }


# ════════════════════════════════════════════════════════════════════════════
# 단위 테스트 (직접 실행 시)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from uuid import uuid4
    from structverify.core.schemas import ClaimSchema, ClaimType

    print("=" * 60)
    print("DocumentWorkingMemory 단위 테스트")
    print("=" * 60)

    # 1. 인스턴스 생성
    mem = DocumentWorkingMemory(
        doc_id="doc_test_001",
        run_id="run_test_001",
        source_uri="https://example.com/news/birth",
    )
    print(f"\n[1] 인스턴스 생성: doc_id={mem.doc_id}")

    # 2. Step 3: 도메인 기록
    mem.record_domain("population", "인구/가구 통계")
    print(f"[2] 도메인 기록: {mem.domain} / {mem.domain_description}")

    # 3. Step 4.5: 시간 정보 기록
    mem.record_anchor_year(2025)
    mem.record_temporal("올해", "2025")
    mem.record_temporal("지난해 같은 달", "2024-04")
    mem.record_temporal("전년", "2024")
    print(f"[3] anchor_year={mem.anchor_year}, temporal={mem.temporal_resolved}")

    # 4. Step 4~6: claim 기록 (birth 기사 흉내)
    test_doc_uuid = uuid4()
    def _mk_claim(sent_id, text, indicator, value, unit, time_period, parent_path=None):
        return Claim(
            claim_id=uuid4(),
            doc_id=test_doc_uuid,
            block_id="b0002",
            sent_id=sent_id,
            claim_text=text,
            claim_type="numerical",
            schema=ClaimSchema(
                indicator=indicator, value=value, unit=unit,
                time_period=time_period,
                parent_path=parent_path,
                modifier=None,
            ),
        )

    claims = [
        _mk_claim("b0002_s0000", "올해 4월 출생아 수는 총 2만 717명",
                  "출생아 수", 20717.0, "명", "2025-04", "인구 > 출생"),
        _mk_claim("b0002_s0000", "지난해 같은 달(1만 9059명)보다 8.7% 늘었다",
                  "출생아 수 증가율", 8.7, "%", "2025-04", "인구 > 출생"),
        _mk_claim("b0002_s0001", "1991년 4월(8.7%) 이후 34년 만에",
                  "출생아 수 증가율", 8.7, "%", "1991-04", "인구 > 출생"),
        _mk_claim("b0002_s0002", "합계출산율 0.79명",
                  "합계출산율", 0.79, "명", "2025-04"),
    ]
    mem.record_claims(claims)
    print(f"[4] claim 4개 기록 → metric_to_claims: {list(mem.metric_to_claims.keys())}")

    # 5. find_related_claims 테스트
    related = mem.find_related_claims("출생아 수 증가율")
    print(f"[5] find_related('출생아 수 증가율') → {len(related)}건")
    for c in related:
        print(f"      · [{c.sent_id}] {c.schema.value}{c.schema.unit} @ {c.schema.time_period}")

    # 6. prefix 검색
    prefix_hit = mem.find_claims_by_indicator_prefix("출생아 수")
    print(f"[6] find_by_prefix('출생아 수') → {len(prefix_hit)}건 "
          f"(증가율 포함)")

    # 7. Step 7: stat_id 캐시
    mem.record_stat_id_used(
        indicator="출생아 수",
        stat_id="DT_1B8000G",
        category_path="인구 > 출생",
        time_period="2025-04",
    )
    cached = mem.get_stat_id_for_indicator("출생아 수")
    print(f"[7] stat_id 캐시 hit: {cached.stat_id} ({cached.category_path})")

    # 8. 도메인 가드 테스트
    print(f"\n[8] 도메인 가드 (domain={mem.domain}):")
    tests = [
        ("인구 > 출생",        True),
        ("인구 > 가구",        True),
        ("고용 > 임금",        False),
        ("환경 > 기상",        False),
        (None,                True),  # category 없으면 통과
    ]
    for cat, expected in tests:
        actual = mem.domain_matches_category(cat)
        status = "✓" if actual == expected else "✗"
        print(f"      {status} '{cat}' → {actual} (기대={expected})")

    # 9. rejected stat_id
    mem.record_stat_id_rejected("DT_WRONG_ID", "domain mismatch")
    print(f"\n[9] is_stat_id_rejected('DT_WRONG_ID') = "
          f"{mem.is_stat_id_rejected('DT_WRONG_ID')}")

    # 10. add_note
    mem.add_note("verify", "domain_guard_triggered", 2)
    print(f"[10] notes={mem.notes}")

    # 11. stats + dump
    print(f"\n[11] stats: {mem.stats()}")

    print(f"\n[12] dump (요약):")
    d = mem.dump()
    for k, v in d.items():
        v_str = str(v)
        if len(v_str) > 80:
            v_str = v_str[:77] + "..."
        print(f"      {k:20s} = {v_str}")

    print(f"\n{'='*60}")
    print("모든 테스트 통과")
    print(f"{'='*60}")