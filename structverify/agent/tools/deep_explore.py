"""
structverify.agent.tools.deep_explore — P28: row-aware deep exploration helper.

배경:
  KOSIS pgvector catalog은 *표 이름*만 임베딩한다. 표 본문(C2_NM 등)에만 있는
  *세부 항목* keyword (예: "체외 충격파 쇄석술", "치료 가능 사망률")는 cosine 검색
  에서 잘 잡히지 않아 정답 표가 top N 밖으로 밀려난다.

  P21B의 row_preview rerank는 *비슷한 시도*였지만 (a) 한 표당 1 row만 보여줘
  패턴 파악 불가, (b) prompt가 "best 골라"라 LLM이 자기 prior로 오선택,
  (c) 5 candidates × 3 prdSe = 15 동시 KOSIS 호출로 API 폭주 → P25에서 disable.

설계 (P28):
  - top N (기본 3) 후보에 대해 *Y prdSe만, 5 row*씩 sample fetch.
  - LLM에 row 패턴 + claim을 던지고 "이 row가 보이면 더 파볼 가치가 있는 표"를
    *외삽적으로* 추천하게 한다. 단순 "best 고르기"가 아니라 "row 단서를 근거로
    한 reasoning".
  - P20 KOSIS cache hit이면 비용 0. cold 시작 시 최악 3건 호출 (Y만).
  - LLM이 "none" 답하면 호출자에게 신호 → reflect에 query refinement 권유.

호출자:
  - catalog_search.py (T1 — top1 점수 낮을 때 사전 보강)
  - loop.py / reflect (T2 — fetch 실패 후 회복용, force_explore=True 전달)
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from structverify.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExplorationResult:
    """deep_explore 반환 형식."""
    best_table_id: str | None
    """LLM이 추천한 best 표 ID. None이면 모두 부적합 OR LLM 호출 실패."""

    reasoning: str
    """LLM이 제시한 한 줄 이유."""

    none_signal: bool
    """LLM이 명시적으로 "none of these"라 답한 경우 True.
    호출자(loop)는 이 신호를 받아 reflect에 *catalog query 재정의* hint를 줘야 함."""

    previewed_ids: list[str]
    """실제로 preview fetch 성공한 stat_id 목록 (디버깅/로깅용)."""

    used: bool
    """deep_explore가 실제 LLM 호출까지 갔는지 (False면 후보 부족/error로 skip)."""


_PICK_COLS = ("ITM_NM", "C1_NM", "C2_NM", "C3_NM", "C4_NM", "PRD_DE", "DT", "UNIT_NM")


async def _preview_fetch(
    candidate_id: str,
    source: Any,
    workspace: Any,
    rows_per_table: int,
) -> dict | None:
    """한 표의 sample row 가져오기. P20 cache hit 시 즉시 반환."""
    try:
        ev = await source.fetch_evidence(
            candidate_id=candidate_id,
            params={
                "newEstPrdCnt": "1",   # 최신 1 시점만 (한 시점에 다수 row 포함)
                "prdSe": "Y",          # Y만 — M/Q fallback X (KOSIS 부하 ↓)
                "_preview": True,
            },
            workspace=workspace,
        )
        if ev is None:
            return None
        rows = ev.get("rows") or []
        if not rows:
            return None
        # 핵심 컬럼만 + rows_per_table 개로 trim
        sample = []
        for r in rows[:rows_per_table]:
            picked = {k: r.get(k) for k in _PICK_COLS if r.get(k) not in (None, "")}
            if picked:
                sample.append(picked)
        return {
            "stat_name": ev.get("stat_name") or "",
            "rows_count": len(rows),
            "sample_rows": sample,
        }
    except Exception as e:
        logger.debug(f"[deep_explore] preview {candidate_id} 실패: {e}")
        return None


def _build_prompt(
    query: str,
    claim_info: dict,
    candidates_with_preview: list[dict],
) -> str:
    """LLM prompt 생성. "best 고르기"가 아니라 "row 단서로 reasoning"."""
    lines = []
    for i, c in enumerate(candidates_with_preview, start=1):
        cid = c.get("id", "")
        cname = (c.get("name", "") or "").strip()
        score = c.get("score")
        head = f"{i}. [{cid}] {cname}"
        if isinstance(score, (int, float)):
            head += f" (catalog_score={score:.3f})"
        lines.append(head)
        prev = c.get("_preview")
        if prev and prev.get("sample_rows"):
            lines.append(f"   - rows_count={prev.get('rows_count')}")
            for j, row in enumerate(prev["sample_rows"], start=1):
                row_str = ", ".join(f"{k}={v!r}" for k, v in row.items())
                lines.append(f"   - row{j}: {row_str}")
        else:
            lines.append("   - (sample row 없음 — preview 실패 또는 빈 표)")

    return f"""당신은 통계표 *탐색 reviewer*입니다. 사용자가 찾는 *구체 항목*이
어느 표의 row에 들어있을 가능성이 높은지, sample row 패턴을 보고 *외삽적으로* 판단하세요.

[사용자 검색 의도]
- query: {query!r}
- indicator (찾는 지표): {claim_info.get('indicator')!r}
- population (대상 집단/지역): {claim_info.get('population')!r}
- time_period: {claim_info.get('time_period')!r}
- unit: {claim_info.get('unit')!r}

[후보 표 + sample rows]
{chr(10).join(lines)}

[판단 기준]
1. ITM_NM / C1_NM~C4_NM 어느 컬럼에 indicator의 *키워드*가 직접/유사 매칭되는 표 우선.
2. 매칭이 안 보여도 row 분류 체계로 보아 "이 표를 더 깊이 파면 (다른 row에) 해당 항목이 있을 가능성"이 높으면 그 표를 추천. *외삽 OK*.
   예: indicator="체외 충격파 쇄석술 장비" + 어떤 표 sample row가 ITM_NM="CT", "MRI" 등 *의료장비 분류*면 → 이 표 더 깊은 row에 쇄석술 장비도 있을 가능성 ↑.
3. row가 명백히 *다른 도메인*(예: indicator=장비인데 row=인구통계)이면 배제.
4. 어느 표에도 단서가 없으면 best_stat_id를 "none"으로 응답. *억지로 고르지 말 것*.

[응답 형식 — JSON only, 다른 텍스트 금지]
{{
  "best_stat_id": "DT_XXX" or "none",
  "reasoning": "row 단서 기반 한 줄 이유 (외삽 reasoning이면 명시)",
  "confidence": 0.0~1.0
}}
"""


def _parse_response(raw: str, candidate_ids: list[str]) -> tuple[str | None, str, bool]:
    """LLM 응답 파싱. (best_id, reasoning, none_signal)."""
    try:
        m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if not m:
            return None, "", False
        data = json.loads(m.group(0))
    except Exception as e:
        logger.debug(f"[deep_explore] JSON 파싱 실패: {e}")
        return None, "", False

    raw_best = (data.get("best_stat_id") or "").strip()
    reasoning = str(data.get("reasoning") or "").strip()

    # "none" 신호 — case-insensitive
    if raw_best.lower() in ("none", "null", "", "n/a"):
        return None, reasoning, True

    # brackets/quotes strip (P24와 동일한 normalize)
    best = raw_best.strip().strip("[]").strip("'\"").strip()
    if best in candidate_ids:
        return best, reasoning, False

    # substring fallback
    for cid in candidate_ids:
        if cid and (best in cid or cid in best):
            logger.info(f"[deep_explore] best={best!r} → substring 매칭 {cid!r}")
            return cid, reasoning, False

    logger.info(f"[deep_explore] best={best!r}가 후보 list에 없음 — 무효 처리")
    return None, reasoning, False


async def deep_explore(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    claim: Any,
    source: Any,
    workspace: Any,
    config: dict | None,
) -> ExplorationResult:
    """top N 후보에 sample row preview + LLM reasoning으로 best table 추천.

    호출자가 trigger 조건 (T1/T2) 검사 후 호출. 본 함수 자체는 무조건 실행.

    Args:
        query: 원본 catalog 검색 쿼리 (LLM에 전달).
        candidates: catalog 검색 결과 후보 list (이미 점수순 정렬됨).
                     각 dict는 최소 {"id": str, "name": str, "score": float}.
        claim: 현재 처리 중인 Claim 객체 (claim.schema 추출용).
        source: BaseDataSource. fetch_evidence(candidate_id, params={_preview:True}) 지원.
        workspace: P20 KOSIS cache 활용.
        config: 전체 config dict — catalog_search.deep_explore.* + agent.llm.* 사용.

    Returns:
        ExplorationResult.
    """
    _cfg = (config or {}).get("catalog_search") or {}
    _dx = _cfg.get("deep_explore") or {}
    top_n = int(_dx.get("top_n") or 3)
    rows_per_table = int(_dx.get("rows_per_table") or 5)

    if not candidates:
        return ExplorationResult(None, "", False, [], used=False)

    top_candidates = candidates[:top_n]
    candidate_ids = [c.get("id", "") for c in top_candidates if c.get("id")]

    if not candidate_ids:
        return ExplorationResult(None, "", False, [], used=False)

    # 1) sample row preview (병렬, KOSIS 부하 고려 — gather)
    preview_tasks = [
        _preview_fetch(cid, source, workspace, rows_per_table)
        for cid in candidate_ids
    ]
    preview_results = await asyncio.gather(*preview_tasks, return_exceptions=False)

    previewed_ids: list[str] = []
    enriched: list[dict[str, Any]] = []
    for c, prev in zip(top_candidates, preview_results):
        d = dict(c)
        if prev is not None:
            d["_preview"] = prev
            previewed_ids.append(c.get("id", ""))
        enriched.append(d)

    if not previewed_ids:
        logger.info("[deep_explore] preview 0건 — LLM 호출 skip")
        return ExplorationResult(None, "", False, [], used=False)

    # 2) claim info 추출
    _schema = getattr(claim, "schema", None) if claim is not None else None
    claim_info = {
        "indicator": (getattr(_schema, "indicator", None) or "") if _schema else "",
        "population": (getattr(_schema, "population", None) or "") if _schema else "",
        "time_period": (getattr(_schema, "time_period", None) or "") if _schema else "",
        "unit": (getattr(_schema, "unit", None) or "") if _schema else "",
    }

    # 3) LLM 호출 — model_tier는 config.catalog_search.deep_explore.model_tier
    prompt = _build_prompt(query, claim_info, enriched)
    model_tier = str(_dx.get("model_tier") or "light").strip().lower()

    from structverify.utils.llm_client import LLMClient
    llm = LLMClient(config=(config or {}).get("llm") or {})
    try:
        raw = await llm.generate(
            prompt=prompt,
            system_prompt=(
                "KOSIS 통계표 탐색 reviewer. row 패턴 기반 외삽 reasoning. JSON만 응답."
            ),
            model_tier=model_tier,
        )
    except Exception as e:
        logger.warning(f"[deep_explore] LLM 호출 실패 (model_tier={model_tier}): {e}")
        return ExplorationResult(None, "", False, previewed_ids, used=False)

    # 4) 파싱 + 결과
    best, reasoning, none_signal = _parse_response(raw, candidate_ids)
    if none_signal:
        logger.info(
            f"[deep_explore] LLM none_signal — reasoning={reasoning[:120]!r}"
        )
    elif best:
        logger.info(
            f"[deep_explore] LLM 추천 best={best!r} reasoning={reasoning[:120]!r}"
        )
    else:
        logger.info(
            f"[deep_explore] LLM 응답 파싱 결과 best=None (none_signal=False) — "
            f"무효 응답으로 처리"
        )

    return ExplorationResult(
        best_table_id=best,
        reasoning=reasoning,
        none_signal=none_signal,
        previewed_ids=previewed_ids,
        used=True,
    )
