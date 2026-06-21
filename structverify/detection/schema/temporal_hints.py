"""detection/schema/temporal_hints.py — schema induction 시점 hint 빌더.

schema_inductor.py에서 분리 (로직 move-only, 동작 변경 없음).

[v6.15] 상대 시점 표현 → 절대 연도 변환표
[v6.19] multi_temporal 판정 — 단정 hint vs 변환표 분기
[v6.20] claim_text 정규식 스캔 — 그래프 누락 보완
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from structverify.core.schemas import Claim
from structverify.utils.logger import get_logger

if TYPE_CHECKING:
    from structverify.graph.graph_builder import ClaimGraph

logger = get_logger(__name__)


# [v6.20] claim 문장 텍스트에서 직접 셀 상대/절대 시점 표현 패턴.
# document_graph의 LLM temporal agent가 한 문장의 표현을 일부 누락하면
# (예: "작년...재작년..."에서 "작년"을 빠뜨림) count_temporal_expressions가
# 1을 반환 → multi_temporal=False → 잘못된 단정 hint. 그래서 그래프와
# 별개로 claim_text를 정규식으로 스캔해 보수적으로 multi 여부를 판정한다.
_TEMPORAL_TEXT_PATTERNS = (
    "재작년", "지지난해", "지지난 해",
    "작년", "지난해", "지난 해", "전년",
    "올해", "금년", "이번 해",
    "내년", "이듬해", "다음 해",
    "내후년",
)


def _count_temporal_in_text(text: str) -> int:
    """claim 문장 텍스트에서 상대 시점 표현의 개수를 센다.

    "작년 X도, 재작년 Y도" → 2 (작년 1 + 재작년 1).
    겹치는 패턴 중복 카운트를 막기 위해, 긴 패턴부터 매칭하며
    매칭된 구간을 소거한다 ('재작년'을 먼저 잡아야 '작년'이
    그 안에서 다시 안 잡힌다).
    """
    if not text:
        return 0
    s = str(text)
    count = 0
    # 긴 패턴 우선 (재작년 → 작년 순서 보장)
    for pat in sorted(_TEMPORAL_TEXT_PATTERNS, key=len, reverse=True):
        while pat in s:
            count += 1
            s = s.replace(pat, "\x00" * len(pat), 1)  # 매칭 구간 소거
    return count


def _build_temporal_hint(graph: "ClaimGraph", claim: Claim) -> str:
    """
    그래프 시점 해소 결과를 prompt hint 텍스트로.

    [v6.15] 상대 시점 표현 매핑 강화:
      anchor_year 기준으로 '내년/올해/작년/지난해/재작년'을 모두 절대 연도로
      변환하는 표를 LLM에게 명시 → time_period=null 방지.
    """
    prov = graph.temporal_provenance(claim)
    anchor_year = graph.get_anchor_year()

    # [v6.19] 한 문장에 시간표현이 여러 개면 (예: "작년 X도, 재작년 Y도")
    # temporal_provenance가 어느 표현이 이 claim의 것인지 구분 못 하고
    # 첫 번째를 무조건 반환한다 → 단정적 hint가 틀릴 수 있음.
    # 이 경우 단정하지 말고 anchor 변환표만 줘서 LLM이 claim 문맥으로
    # 직접 시점을 고르게 한다.
    _te_count = graph.count_temporal_expressions(claim)
    # [v6.20] 그래프 카운트와 별개로 claim 문장 텍스트도 직접 스캔.
    # temporal agent가 표현을 누락해도(그래프 te_count=1) 텍스트에
    # 상대표현이 2개 이상이면 multi로 판정 → 잘못된 단정 hint 방지.
    _text_te_count = _count_temporal_in_text(
        getattr(claim, "claim_text", "") or ""
    )
    multi_temporal = (_te_count > 1) or (_text_te_count > 1)

    # [v6.19 진단] multi_temporal 판정과 분기 결정을 로그로 — 어느 경로를
    # 탔는지 안 보여서 temporal 수정이 먹혔는지 확인이 안 됨.
    _branch = (
        "단정(prov)" if (prov and prov.get("resolved") and not multi_temporal)
        else ("변환표(anchor)" if anchor_year is not None else "없음")
    )
    logger.info(
        f"[temporal_hint] {getattr(claim, 'sent_id', '?')}: "
        f"te_count={_te_count} text_te={_text_te_count} "
        f"multi_temporal={multi_temporal} "
        f"prov_resolved={(prov or {}).get('resolved')} "
        f"prov_expr={(prov or {}).get('expression')!r} "
        f"anchor={anchor_year} → branch={_branch}"
    )

    if prov and prov.get("resolved") and not multi_temporal:
        return (
            f"\n[시점 정보 — 그래프 해소 결과]\n"
            f"- 원문 표현: {prov.get('expression')}\n"
            f"- 해소된 절대 시점: {prov['resolved']}\n"
            f"- 근거: {prov.get('basis') or '문서 anchor 기반'}\n"
            f"위 절대 시점을 time_period로 사용하세요."
        )
    elif anchor_year is not None:
        # [v6.15] 상대 표현 → 절대 연도 변환표를 명시적으로 제공
        multi_note = ""
        if multi_temporal:
            # [v6.19] 한 문장에 시간표현이 여러 개 — 수치별로 구분 지시
            multi_note = (
                f"- ⚠️ 이 문장에는 시점 표현이 *둘 이상* 있습니다 "
                f"(예: '작년 X도, 재작년 Y도').\n"
                f"  각 수치 바로 앞/근처의 시점 표현을 보고 schema마다 "
                f"time_period를 *개별적으로* 정확히 매칭하세요.\n"
                f"  모든 수치에 같은 시점을 쓰지 마세요.\n"
            )
        return (
            f"\n[시점 정보 — 문서 anchor]\n"
            f"- 이 문서의 기준 연도(anchor_year): {anchor_year}\n"
            f"{multi_note}"
            f"- 상대 시점 표현은 *반드시* 아래 표대로 절대 연도로 변환하세요:\n"
            f"    '내후년'        → {anchor_year + 2}\n"
            f"    '내년/이듬해'   → {anchor_year + 1}\n"
            f"    '올해/금년/현재' → {anchor_year}\n"
            f"    '작년/지난해'   → {anchor_year - 1}\n"
            f"    '재작년'        → {anchor_year - 2}\n"
            f"- ★ 검증 대상 문장에 위 상대 표현이 하나라도 있으면\n"
            f"  time_period를 절대 연도(예: '{anchor_year + 1}')로 *반드시* 채우세요.\n"
            f"- ★ time_period를 null로 두지 마세요. 시점 단서가 전혀 없을 때만 null."
        )
    return ""
