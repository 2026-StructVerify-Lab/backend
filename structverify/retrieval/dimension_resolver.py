"""
structverify.retrieval.dimension_resolver — P34: KOSIS 표 차원(itmId/objL) 동적 결정.

배경:
  KOSIS Open API는 표마다 *N차원 구조* (ITM × OBJL01 × OBJL02 × ... × PRD).
  호출 시 itmId/objL1/objL2...를 *특정 코드*로 지정하면 그 *슬라이스*만 받음.
  지정 안 하면 connector가 getMeta(CMMT)[0]의 ITM_ID/OBJ_ID(보통 *합계 코드*)를
  자동 박아 *합계 row만* 받음 → 세부 항목 row가 *아예 안 옴*.

  예 DT_HIRA4Q (3차원: 시군구 × 장비종류 × 분기):
    - getMeta(CMMT)[0].ITM_ID = '00' (진단방사선·특수의료장비 합계)
    - 호출에 itmId='00' 박힘 → 세부 ITM (체외 충격파 쇄석술기, CT, MRI...) 누락
    - row 매칭에서 "체외 충격파 쇄석술 장비" indicator 찾을 row 0건

설계:
  1) fetch 전에 getMeta(ITM) + getMeta(OBJL01~08) 호출 (병렬, ~1초)
  2) 각 차원의 (코드, 이름) pair list를 LLM에 노출
  3) LLM이 claim의 indicator/population과 *의미적 매칭*되는 코드 선택
  4) {itmId, objL1, ...} dict 반환
  5) fetch 호출 시 그 코드들을 params에 박음 → *정확한 슬라이스*만 받음

  매칭 실패 시 None 반환 → 호출자가 기존 동작 (cmmt_rows[0]) 또는 fallback.

캐시:
  per-job + per-stat_id cache (workspace에 저장 안 함, 메모리만). 같은 stat_id로
  다시 호출돼도 LLM/API 1회만.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from structverify.utils.logger import get_logger

logger = get_logger(__name__)


# process-level cache: {stat_id: {itmId, objL1, ...}} (또는 None for "매칭 실패")
_DIM_CACHE: dict[str, dict[str, str] | None] = {}


def _extract_pairs(meta_rows: Any, code_keys: tuple[str, ...], name_keys: tuple[str, ...]) -> list[tuple[str, str]]:
    """KOSIS getMeta 응답에서 (code, name) pair list 추출.

    응답 형식이 *type마다 다름*:
      - ITM 응답: [{ITM_ID, ITM_NM, ...}, ...]
      - OBJL01 응답: [{OBJ_ID, OBJ_NM, ...}, ...] 또는 [{C1, C1_NM, ...}]
    """
    if not isinstance(meta_rows, list):
        return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for r in meta_rows:
        if not isinstance(r, dict):
            continue
        code = ""
        for k in code_keys:
            v = r.get(k)
            if v:
                code = str(v).strip()
                break
        name = ""
        for k in name_keys:
            v = r.get(k)
            if v:
                name = str(v).strip()
                break
        if code and code not in seen:
            seen.add(code)
            out.append((code, name or code))
    return out


def _build_prompt(
    indicator: str,
    population: str,
    claim_text: str,
    parent_path: str,
    stat_name: str,
    itm_pairs: list[tuple[str, str]],
    obj_pairs_by_level: dict[int, list[tuple[str, str]]],
) -> str:
    """LLM prompt — 각 차원의 코드 list를 보여주고 매칭 코드 선택."""
    lines: list[str] = []
    if itm_pairs:
        shown = itm_pairs[:80]
        more = f" (외 {len(itm_pairs)-len(shown)}개)" if len(itm_pairs) > len(shown) else ""
        pairs_str = ", ".join(f'{code!r}→{name!r}' for code, name in shown)
        lines.append(f"### ITM (통계항목, 코드→이름):\n{pairs_str}{more}")
    for level in sorted(obj_pairs_by_level.keys()):
        pairs = obj_pairs_by_level[level]
        if not pairs:
            continue
        shown = pairs[:80]
        more = f" (외 {len(pairs)-len(shown)}개)" if len(pairs) > len(shown) else ""
        pairs_str = ", ".join(f'{code!r}→{name!r}' for code, name in shown)
        lines.append(f"### OBJL{level:02d} (분류 {level}차, 코드→이름):\n{pairs_str}{more}")

    dim_block = "\n\n".join(lines) if lines else "(차원 정보 없음)"

    return f"""당신은 KOSIS 통계표의 *N차원 슬라이스 코드*를 결정하는 reviewer입니다.
사용자 claim에 맞는 *itmId / objL1~objL8*을 각 차원의 코드 list에서 선택하세요.

[사용자 검색 의도]
- 표: {stat_name!r}
- indicator (찾는 지표): {indicator!r}
- population (대상 집단/지역): {population!r}
- claim 원문: {claim_text or '(없음)'}
- parent_path: {parent_path or '(없음)'}

[표의 차원 정보 — getMeta 응답]
{dim_block}

[판단 기준]
1. ITM 차원: indicator의 *핵심 키워드*와 의미적으로 매칭되는 ITM_ID 1개 선택.
   - 예: indicator="체외 충격파 쇄석술 장비" → ITM_NM에 "체외 충격파 쇄석술기" 또는 "ESWL" 있으면 그 ITM_ID.
   - *상위 합계 코드* (예: ITM_NM='진단방사선·특수의료장비' 같은 *전체 묶음*)는 피하기.
   - 매칭 없으면 itmId를 "ALL"로 (모든 항목 받기).
2. OBJL01 차원: population의 지역/집단과 매칭되는 코드 1개 선택.
   - 예: population="강원도" → OBJ_NM에 "강원" 또는 "강원도" 있으면 그 코드.
   - 매칭 없으면 "ALL".
3. OBJL02 이상: 보통 ALL 권장 (세부 분류는 다 받기). 단 특정 분류 매칭이 명확하면 그 코드.
4. 매칭이 확실하지 *않으면* 해당 차원은 "ALL" — 추측 금지.

[응답 형식 — JSON only, 다른 텍스트 금지]
{{
  "itmId": "<코드> 또는 ALL",
  "objL1": "<코드> 또는 ALL",
  "objL2": "<코드> 또는 ALL",
  "reasoning": "한 줄 이유 — 어느 차원의 어떤 값에 매칭했는지"
}}

* 표에 없는 차원은 응답에서 빼도 됨. 모두 ALL인 경우도 valid.
"""


def _parse(raw: str) -> dict[str, str]:
    """LLM 응답 파싱. {itmId, objL1, ...} dict 반환. 빈 dict면 실패."""
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {}
        data = json.loads(m.group(0))
    except Exception as e:
        logger.debug(f"[dimension_resolver] 파싱 실패: {e}")
        return {}

    out: dict[str, str] = {}
    for key in ("itmId", "objL1", "objL2", "objL3", "objL4", "objL5", "objL6", "objL7", "objL8"):
        v = data.get(key)
        if v is None:
            continue
        s = str(v).strip().strip("'\"").strip()
        if s:
            out[key] = s
    return out


async def resolve_dimensions(
    *,
    stat_id: str,
    stat_name: str,
    source: Any,
    indicator: str,
    population: str,
    claim_text: str,
    parent_path: str,
    config: dict | None,
    use_cache: bool = True,
) -> dict[str, str] | None:
    """KOSIS 표의 차원 메타를 받아 LLM이 claim 매칭 코드 결정.

    Args:
        stat_id: KOSIS 표 ID.
        stat_name: 표 이름 (LLM 프롬프트용).
        source: BaseDataSource (get_table_meta 지원해야).
        indicator/population/claim_text/parent_path: claim 컨텍스트.
        config: 전체 config dict.
        use_cache: process-level cache 사용 여부.

    Returns:
        {itmId, objL1, ...} dict (값은 KOSIS 코드 또는 "ALL"). 매칭 실패/메타 없음 시 None.
    """
    if not stat_id:
        return None

    # 1) cache hit?
    if use_cache and stat_id in _DIM_CACHE:
        cached = _DIM_CACHE[stat_id]
        if cached is None:
            logger.info(f"[dimension_resolver] cache hit (negative): {stat_id}")
            return None
        logger.info(f"[dimension_resolver] cache hit: {stat_id} → {cached}")
        return cached

    # 2) 메타 병렬 호출 — ITM + OBJL01~OBJL04 (5개까지만 시도, 표 대부분 ≤3차원)
    meta_tasks = {
        "ITM": source.get_table_meta(stat_id, "ITM"),
        "OBJL01": source.get_table_meta(stat_id, "OBJL01"),
        "OBJL02": source.get_table_meta(stat_id, "OBJL02"),
        "OBJL03": source.get_table_meta(stat_id, "OBJL03"),
        "OBJL04": source.get_table_meta(stat_id, "OBJL04"),
    }
    results = await asyncio.gather(*meta_tasks.values(), return_exceptions=False)
    metas = dict(zip(meta_tasks.keys(), results))

    # 3) (code, name) pair 추출
    itm_pairs = _extract_pairs(
        metas.get("ITM"),
        code_keys=("ITM_ID",),
        name_keys=("ITM_NM",),
    )
    obj_pairs_by_level: dict[int, list[tuple[str, str]]] = {}
    for level in range(1, 5):
        key = f"OBJL{level:02d}"
        meta = metas.get(key)
        # OBJ 응답은 OBJ_ID/OBJ_NM 또는 C{level}/C{level}_NM 둘 다 가능 — 우선순위 시도
        pairs = _extract_pairs(
            meta,
            code_keys=("OBJ_ID", f"C{level}"),
            name_keys=("OBJ_NM", f"C{level}_NM"),
        )
        if pairs:
            obj_pairs_by_level[level] = pairs

    if not itm_pairs and not obj_pairs_by_level:
        logger.info(f"[dimension_resolver] 메타 비어있음: {stat_id} → 차원 결정 skip")
        if use_cache:
            _DIM_CACHE[stat_id] = None
        return None

    # 4) LLM 호출
    _rg_cfg = ((config or {}).get("kosis") or {}).get("dimension_resolver") or {}
    model_tier = str(_rg_cfg.get("model_tier") or "light").strip().lower()

    prompt = _build_prompt(
        indicator=indicator,
        population=population,
        claim_text=claim_text[:400] if claim_text else "",
        parent_path=parent_path,
        stat_name=stat_name,
        itm_pairs=itm_pairs,
        obj_pairs_by_level=obj_pairs_by_level,
    )
    from structverify.utils.llm_client import LLMClient
    llm = LLMClient(config=(config or {}).get("llm") or {})
    try:
        raw = await llm.generate(
            prompt=prompt,
            system_prompt="KOSIS N차원 슬라이스 코드 reviewer. JSON만 응답.",
            model_tier=model_tier,
        )
    except Exception as e:
        logger.warning(f"[dimension_resolver] LLM 호출 실패: {e}")
        if use_cache:
            _DIM_CACHE[stat_id] = None
        return None

    dims = _parse(raw)
    if not dims:
        logger.info(
            f"[dimension_resolver] 파싱 실패 raw={raw[:200]!r} stat_id={stat_id}"
        )
        if use_cache:
            _DIM_CACHE[stat_id] = None
        return None

    logger.info(
        f"[dimension_resolver] {stat_id} → {dims} "
        f"(ITM 후보 {len(itm_pairs)}개, OBJ {sum(len(v) for v in obj_pairs_by_level.values())}개)"
    )
    if use_cache:
        _DIM_CACHE[stat_id] = dims
    return dims


def clear_cache() -> None:
    """테스트/디버깅용 cache 초기화."""
    _DIM_CACHE.clear()
