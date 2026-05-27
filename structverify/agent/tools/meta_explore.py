"""
structverify.agent.tools.meta_explore — P30: KOSIS getMeta 기반 LLM reasoning.

배경:
  P28 deep_explore (row preview)는 표마다 전체 데이터(8천~1.3만 row)를 받아
  표당 22~26초 소요. KOSIS의 *메타 API* (getMeta type=ITM/OBJL)는 같은 표의
  *항목/분류 list* 만 반환해 표당 ~1초로 훨씬 가벼움. 게다가:

    - ITM_NM list에 "체외 충격파 쇄석술기" 같은 *세부 row keyword*가 직접 포함됨
    - C1_NM/C2_NM 분류에 "강원도" 같은 지역 코드가 명시됨
    → LLM이 *확실한 증거 기반*으로 best 표 식별 가능 (외삽 reasoning 불필요)

deep_explore와 인터페이스 동일 (ExplorationResult 반환). catalog_search Tool이
config.catalog_search.deep_explore.explore_mode = "meta"일 때 이 함수를 호출.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from structverify.utils.logger import get_logger

from .deep_explore import ExplorationResult

logger = get_logger(__name__)


async def _fetch_meta(
    candidate_id: str,
    source: Any,
    include_obj: bool,
) -> dict | None:
    """한 표의 getMeta(ITM) + (옵션) getMeta(OBJL01) 호출.

    Returns:
        {"itm": [...], "obj": [...]} dict. 실패 시 None.
    """
    try:
        itm_task = source.get_table_meta(candidate_id=candidate_id, meta_type="ITM")
        if include_obj:
            obj_task = source.get_table_meta(candidate_id=candidate_id, meta_type="OBJL01")
            itm, obj = await asyncio.gather(itm_task, obj_task)
        else:
            itm = await itm_task
            obj = None
        if itm is None and obj is None:
            return None
        return {"itm": itm or [], "obj": obj or []}
    except Exception as e:
        logger.debug(f"[meta_explore] meta fetch {candidate_id} 실패: {e}")
        return None


def _extract_names(meta_rows: Any, name_keys: tuple[str, ...]) -> list[str]:
    """KOSIS getMeta 응답에서 *이름* list 추출.

    응답은 보통 [{"ITM_ID": "...", "ITM_NM": "...", ...}, ...] 또는
    [{"C1_NM": "...", "C1": "...", ...}] 형식. name_keys 순서대로 첫 hit 사용.
    """
    if not isinstance(meta_rows, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for r in meta_rows:
        if not isinstance(r, dict):
            continue
        for k in name_keys:
            v = r.get(k)
            if v and isinstance(v, str):
                s = v.strip()
                if s and s not in seen:
                    seen.add(s)
                    out.append(s)
                break
    return out


def _build_prompt(
    query: str,
    claim_info: dict,
    candidates_with_meta: list[dict],
    max_items_per_table: int = 50,
) -> str:
    """LLM prompt — ITM list + OBJ list 보고 best 표 선택."""
    lines: list[str] = []
    for i, c in enumerate(candidates_with_meta, start=1):
        cid = c.get("id", "")
        cname = (c.get("name", "") or "").strip()
        score = c.get("score")
        head = f"{i}. [{cid}] {cname}"
        if isinstance(score, (int, float)):
            head += f" (catalog_score={score:.3f})"
        lines.append(head)
        meta = c.get("_meta")
        if meta:
            itm_names = _extract_names(meta.get("itm"), ("ITM_NM",))
            obj_names = _extract_names(meta.get("obj"), ("C1_NM", "OBJL_NM"))
            if itm_names:
                shown = itm_names[:max_items_per_table]
                more = f" (외 {len(itm_names)-len(shown)}개)" if len(itm_names) > len(shown) else ""
                lines.append(f"   - 통계항목 ITM_NM: {', '.join(shown)}{more}")
            if obj_names:
                shown = obj_names[:max_items_per_table]
                more = f" (외 {len(obj_names)-len(shown)}개)" if len(obj_names) > len(shown) else ""
                lines.append(f"   - 분류 OBJ_NM: {', '.join(shown)}{more}")
            if not itm_names and not obj_names:
                lines.append("   - (메타 비어있음)")
        else:
            lines.append("   - (메타 fetch 실패 — catalog score만으로 추정)")

    return f"""당신은 통계표 *식별 reviewer*입니다. 사용자가 찾는 *구체 항목*이
어느 표의 통계항목(ITM_NM) 또는 분류(OBJ_NM)에 포함되어 있는지 판단하세요.

[사용자 검색 의도]
- query: {query!r}
- indicator (찾는 지표): {claim_info.get('indicator')!r}
- population (대상 집단/지역): {claim_info.get('population')!r}
- time_period: {claim_info.get('time_period')!r}
- unit: {claim_info.get('unit')!r}

[후보 표 + 메타 항목/분류]
{chr(10).join(lines)}

[판단 기준]
1. 어느 표의 ITM_NM list에 indicator의 *핵심 키워드*가 직접/유사 매칭되는가? (예: indicator="체외 충격파 쇄석술 장비" → ITM_NM에 "체외 충격파 쇄석술기" 또는 "ESWL" 같은 항목이 있으면 매칭).
2. OBJ_NM list가 population(지역/집단)을 포함하는가? (예: population="강원도" → OBJ에 "강원" 또는 시도 분류 있으면 매칭).
3. ITM+OBJ 둘 다 매칭되는 표가 정답일 가능성 최상.
4. 어느 표에도 매칭 단서가 없으면 best_stat_id를 "none"으로 응답. *억지로 고르지 말 것*.

[응답 형식 — JSON only, 다른 텍스트 금지]
{{
  "best_stat_id": "DT_XXX" or "none",
  "reasoning": "ITM/OBJ 매칭 근거 한 줄 (어떤 항목이 어디 있는지 명시)",
  "confidence": 0.0~1.0
}}
"""


def _parse_response(raw: str, candidate_ids: list[str]) -> tuple[str | None, str, bool]:
    """LLM 응답 파싱 (deep_explore와 동일 로직)."""
    try:
        m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if not m:
            return None, "", False
        data = json.loads(m.group(0))
    except Exception as e:
        logger.debug(f"[meta_explore] JSON 파싱 실패: {e}")
        return None, "", False

    raw_best = (data.get("best_stat_id") or "").strip()
    reasoning = str(data.get("reasoning") or "").strip()

    if raw_best.lower() in ("none", "null", "", "n/a"):
        return None, reasoning, True

    best = raw_best.strip().strip("[]").strip("'\"").strip()
    if best in candidate_ids:
        return best, reasoning, False

    for cid in candidate_ids:
        if cid and (best in cid or cid in best):
            logger.info(f"[meta_explore] best={best!r} → substring 매칭 {cid!r}")
            return cid, reasoning, False

    logger.info(f"[meta_explore] best={best!r}가 후보 list에 없음 — 무효 처리")
    return None, reasoning, False


async def meta_explore(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    claim: Any,
    source: Any,
    workspace: Any,
    config: dict | None,
) -> ExplorationResult:
    """top N 표의 getMeta(ITM/OBJ) → LLM이 *항목 list* 보고 best 표 식별.

    deep_explore와 동일 인터페이스. config.catalog_search.deep_explore의
    top_n / model_tier / include_obj 사용.

    Args:
        query: catalog 검색 쿼리.
        candidates: catalog 후보 list (이미 점수순). 각 dict는 {"id", "name", "score"}.
        claim: Claim 객체.
        source: BaseDataSource (get_table_meta 지원해야).
        workspace: (현재 미사용, 인터페이스 호환용).
        config: 전체 config dict.

    Returns:
        ExplorationResult.
    """
    _cfg = (config or {}).get("catalog_search") or {}
    _dx = _cfg.get("deep_explore") or {}
    top_n = int(_dx.get("top_n") or 5)
    include_obj = bool(_dx.get("include_obj", True))

    if not candidates:
        return ExplorationResult(None, "", False, [], used=False)

    top_candidates = candidates[:top_n]
    candidate_ids = [c.get("id", "") for c in top_candidates if c.get("id")]
    if not candidate_ids:
        return ExplorationResult(None, "", False, [], used=False)

    # 1) meta 병렬 fetch
    meta_tasks = [_fetch_meta(cid, source, include_obj) for cid in candidate_ids]
    meta_results = await asyncio.gather(*meta_tasks, return_exceptions=False)

    previewed_ids: list[str] = []
    enriched: list[dict[str, Any]] = []
    for c, meta in zip(top_candidates, meta_results):
        d = dict(c)
        if meta is not None:
            d["_meta"] = meta
            previewed_ids.append(c.get("id", ""))
        enriched.append(d)

    if not previewed_ids:
        logger.info("[meta_explore] 메타 fetch 0건 — LLM 호출 skip")
        return ExplorationResult(None, "", False, [], used=False)

    # 2) claim info
    _schema = getattr(claim, "schema", None) if claim is not None else None
    claim_info = {
        "indicator": (getattr(_schema, "indicator", None) or "") if _schema else "",
        "population": (getattr(_schema, "population", None) or "") if _schema else "",
        "time_period": (getattr(_schema, "time_period", None) or "") if _schema else "",
        "unit": (getattr(_schema, "unit", None) or "") if _schema else "",
    }

    # 3) LLM 호출
    prompt = _build_prompt(query, claim_info, enriched)
    model_tier = str(_dx.get("model_tier") or "light").strip().lower()

    from structverify.utils.llm_client import LLMClient
    llm = LLMClient(config=(config or {}).get("llm") or {})
    try:
        raw = await llm.generate(
            prompt=prompt,
            system_prompt=(
                "KOSIS 통계표 식별 reviewer. ITM/OBJ 메타 기반 reasoning. JSON만 응답."
            ),
            model_tier=model_tier,
        )
    except Exception as e:
        logger.warning(f"[meta_explore] LLM 호출 실패 (model_tier={model_tier}): {e}")
        return ExplorationResult(None, "", False, previewed_ids, used=False)

    best, reasoning, none_signal = _parse_response(raw, candidate_ids)
    if none_signal:
        logger.info(f"[meta_explore] LLM none_signal — reasoning={reasoning[:120]!r}")
    elif best:
        logger.info(f"[meta_explore] LLM 추천 best={best!r} reasoning={reasoning[:120]!r}")
    else:
        logger.info(f"[meta_explore] LLM 응답 파싱 결과 best=None — 무효 응답으로 처리")

    return ExplorationResult(
        best_table_id=best,
        reasoning=reasoning,
        none_signal=none_signal,
        previewed_ids=previewed_ids,
        used=True,
    )
