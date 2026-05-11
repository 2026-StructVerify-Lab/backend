"""
core/memory.py — DocumentMemory (v6.4 신규)

[배경]
한 문서 내 claim들은 순차 처리되며, 후속 claim의 해석에 앞선 claim들의 결과가
큰 도움이 된다:
  s0000: "올 4월 출생아 6.7% 증가"   → indicator=출생아 수, ratio
  s0001: "이는 1991년 4월(8.1%) 이후 …" → "이는"이 s0000을 가리킴
  s0002: "올 4월 합계출산율도 0.04명 증가" → 같은 시점 패턴
  s0003: "올 4월 혼인 건수 3.9% 증가"  → 같은 시점 패턴

또 KOSIS 조회는 비싼 호출이라 (catalog 검색 + LLM agent select + fetch),
같은 통계표/시점을 여러 claim이 참조하면 캐싱해야 한다.

[설계]
DocumentMemory는 RuntimeAgent.process() 한 번 호출 동안만 살아있음.
호출이 끝나면 폐기 — 라이브러리 사용자에겐 보이지 않음.

세 가지 누적 정보 보관:
1. processed_claims : 순서대로 처리된 claim들의 핵심 정보
   → schema_inductor가 다음 claim의 LLM prompt에 "이전 claim 컨텍스트" 주입
2. table_cache      : (stat_id, time_period, population) → StatData
   → 같은 표/시점 재조회 방지 + 일관성 유지
3. last_stat_for_indicator : indicator → 마지막으로 성공한 stat_id
   → catalog 검색을 건너뛰고 직접 fetch (옵션)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from structverify.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessedClaimMemo:
    """이전에 처리된 한 claim의 핵심 정보 (다음 claim의 prompt에 주입용)."""
    sent_id: str
    claim_text: str
    indicator: str | None
    value: float | None
    unit: str | None
    time_period: str | None
    value_role: str
    combiner: str
    requirements_summary: str  # 예: "endpoint_a@2025-04, endpoint_b@2024-04"
    verdict: str | None = None
    evidence_stat_id: str | None = None
    # [v6.4] 두 시점 fetch에 성공한 endpoint 값들 — 다음 claim이 "이는"으로 참조할 때 사용
    computed_value: float | None = None  # combiner 결과
    endpoint_a_value: float | None = None
    endpoint_b_value: float | None = None


@dataclass
class DocumentMemory:
    """
    한 문서 처리 중 누적되는 작업 기억.

    runtime_agent.process() 시작 시 새로 생성, 매 claim 처리 후 update.
    schema_inductor / kosis_connector / evidence_check가 옵셔널 인자로 받음.
    """
    # 처리 완료된 claim들 (순서대로)
    processed_claims: list[ProcessedClaimMemo] = field(default_factory=list)

    # KOSIS 표 단위 캐시: (stat_id, prd_de_normalized, population) → StatData-like
    table_cache: dict[tuple[str, str, str], Any] = field(default_factory=dict)

    # indicator → 마지막으로 성공한 stat_id
    last_stat_for_indicator: dict[str, str] = field(default_factory=dict)

    # ── 누적 claim 컨텍스트 ───────────────────────────────────────────────
    def append_processed(self, memo: ProcessedClaimMemo) -> None:
        self.processed_claims.append(memo)

    def recent_context_for_prompt(self, max_items: int = 4) -> str:
        """
        schema_inductor prompt에 주입할 "이전 claim 처리 요약" 텍스트.
        최근 max_items개만 — 너무 길면 토큰 낭비 + LLM 산만.
        """
        if not self.processed_claims:
            return ""

        recent = self.processed_claims[-max_items:]
        lines = ["[이전 처리된 claim들 (참고)]"]
        for memo in recent:
            val_repr = (
                f"{memo.value}{memo.unit or ''}" if memo.value is not None else "?"
            )
            line = (
                f"  · {memo.sent_id}: indicator={memo.indicator!r}, "
                f"value={val_repr}, time={memo.time_period!r}, "
                f"role={memo.value_role}, combiner={memo.combiner}"
            )
            lines.append(line)
            if memo.requirements_summary:
                lines.append(f"      plan: {memo.requirements_summary}")
            # [v6.4] 계산된 값도 노출 — 다음 claim이 "이는"으로 가리킬 수 있음
            if memo.computed_value is not None:
                lines.append(
                    f"      계산값={memo.computed_value:+.3f} "
                    f"(endpoint_a={memo.endpoint_a_value}, "
                    f"endpoint_b={memo.endpoint_b_value})"
                )
        lines.append(
            "위는 같은 문서의 앞 문장들이 만든 schema/plan입니다. "
            '현재 문장에 "이는", "같은", "전년" 같은 지시/대명사가 있으면 '
            "앞 claim들과의 관계를 추론하는 데 사용하세요. "
            "단, 단순 복제는 금지 — 현재 문장의 실제 수치/시점을 그대로 따르세요."
        )
        return "\n".join(lines)

    # ── KOSIS 표 캐시 ─────────────────────────────────────────────────────
    @staticmethod
    def _normalize_time_key(time_period: str | None) -> str:
        if not time_period:
            return ""
        # "2024-04" → "202404", "2024" → "2024"
        return time_period.replace("-", "").strip()

    def get_cached_table(
        self,
        stat_id: str,
        time_period: str | None,
        population: str | None,
    ) -> Any | None:
        key = (
            stat_id,
            self._normalize_time_key(time_period),
            (population or "").strip(),
        )
        hit = self.table_cache.get(key)
        if hit is not None:
            logger.info(
                f"[memory] KOSIS 캐시 hit: [{stat_id}] {time_period} ({population})"
            )
        return hit

    def cache_table(
        self,
        stat_id: str,
        time_period: str | None,
        population: str | None,
        data: Any,
    ) -> None:
        key = (
            stat_id,
            self._normalize_time_key(time_period),
            (population or "").strip(),
        )
        self.table_cache[key] = data

    # ── 디버그 ────────────────────────────────────────────────────────────
    def summary(self) -> str:
        return (
            f"DocumentMemory(processed={len(self.processed_claims)}, "
            f"table_cache={len(self.table_cache)})"
        )
