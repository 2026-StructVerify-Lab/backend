"""
structverify.retrieval.row_matcher — P33c: LLM 기반 row indicator 매칭.

배경:
  _select_best_row()의 _ind_match()는 정규화 후 정확 일치/substring 양방향 비교
  하는 룰베이스. KOSIS 표가 *계층적 분류*를 가져서 row의 ITM_NM이 *상위 카테고리*
  (예: "진단방사선·특수의료장비")인데 사용자 indicator가 *세부 항목*(예:
  "체외 충격파 쇄석술 장비")이면 직접 매칭이 안 됨.

  LLM이 rows의 unique ITM_NM / C1_NM~C4_NM 값들을 보고 indicator와
  *의미적으로 매칭*되는 컬럼 값을 식별해 그 값으로 row 필터링.

용도:
  _select_best_row()의 indicator pre-filter가 0건일 때만 호출 (룰 통과 시엔
  비용 절감 위해 LLM 호출 X). config.kosis.relevance_guard.row_match_llm_fallback
  이 true일 때만 활성화.

반환:
  매칭 row list (1개 이상) 또는 빈 list. 빈 list면 진짜 그 표에 indicator가
  없는 것이므로 호출자가 None 반환.
"""
from __future__ import annotations

import json
import re
from typing import Any

from structverify.utils.logger import get_logger

logger = get_logger(__name__)


_INDICATOR_FIELDS = ("ITM_NM", "C1_NM", "C2_NM", "C3_NM", "C4_NM")


def _extract_unique_values(rows: list[dict], field: str, max_n: int = 80) -> list[str]:
    """rows의 특정 컬럼에서 unique 값 list 추출 (LLM에 노출용)."""
    seen: set[str] = set()
    out: list[str] = []
    for r in rows:
        v = r.get(field)
        if not v:
            continue
        s = str(v).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
            if len(out) >= max_n:
                break
    return out


def _build_prompt(
    indicator: str,
    claim_text: str,
    parent_path: str,
    population: str,
    field_values: dict[str, list[str]],
) -> str:
    """LLM prompt — unique 컬럼 값들 보고 indicator와 매칭되는 값 추천."""
    lines = []
    for f, vals in field_values.items():
        if not vals:
            continue
        # 너무 길면 자름
        shown = vals[:50]
        more = f" (외 {len(vals)-len(shown)}개)" if len(vals) > len(shown) else ""
        lines.append(f"- {f}: {', '.join(repr(v) for v in shown)}{more}")
    field_block = "\n".join(lines) if lines else "(분류 컬럼 값 없음)"

    return f"""당신은 KOSIS 통계표 row 매칭 reviewer입니다. 사용자 indicator가
표 안 *어떤 row 분류 값*에 해당하는지 의미적으로 판단하세요.

[사용자 검색 의도]
- indicator: {indicator!r}
- claim 원문: {claim_text or '(없음)'}
- parent_path: {parent_path or '(없음)'}
- population: {population or '(없음)'}

[표 안 row의 분류 컬럼 unique 값 list]
{field_block}

[판단 기준]
1. indicator의 *핵심 키워드*가 어느 컬럼의 어떤 값에 *직접/유사 매칭*되는가? (예: indicator="체외 충격파 쇄석술 장비" → ITM_NM 또는 C2_NM에 "체외 충격파 쇄석술기" 또는 "ESWL" 같은 값이 있으면 그것).
2. parent_path(계층)에 비추어 *상위 카테고리* row (예: "진단방사선·특수의료장비" 합계)는 매칭이 *아니다*. *세부 항목 row*가 필요.
3. claim 원문에 "보유대수", "장비 수" 같은 *수량 표현*은 indicator의 일반어이므로 row 분류 값과 직접 비교 불필요.
4. 어느 컬럼에도 indicator와 의미적 매칭이 없으면 → 그 표에 indicator 없는 것 → "matches": [] 반환.

★ field 라벨 정확성:
  - "field"는 *반드시* 위 [표 안 row의 분류 컬럼 unique 값 list]에 *명시된 라벨*과 동일하게 적기.
  - 예: 값이 "ITM_NM" list가 아니라 "C2_NM" list 안에 있으면 → field="C2_NM" (절대 ITM_NM이라 적지 말 것).
  - field 잘못 적으면 row 필터링 0건이 되어 검증 실패. 정확한 컬럼명 사용 필수.

[응답 형식 — JSON only, 다른 텍스트 금지]
{{
  "matches": [{{"field": "<컬럼명>", "value": "<매칭 값>"}}, ...],
  "reasoning": "한 줄 이유"
}}

* matches는 0개 이상. 보통 1~3개. *상위 카테고리* 값은 절대 포함하지 마세요.
"""


def _parse(raw: str) -> tuple[list[dict], str]:
    """LLM 응답 파싱. (matches list, reasoning)."""
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return [], ""
        data = json.loads(m.group(0))
        matches = data.get("matches") or []
        reasoning = str(data.get("reasoning") or "")
        if not isinstance(matches, list):
            return [], reasoning
        out: list[dict] = []
        for m in matches:
            if isinstance(m, dict) and m.get("field") and m.get("value"):
                out.append({"field": str(m["field"]), "value": str(m["value"])})
        return out, reasoning
    except Exception as e:
        logger.debug(f"[row_matcher] 파싱 실패: {e}")
        return [], ""


async def llm_select_rows(
    *,
    rows: list[dict],
    indicator: str,
    claim_text: str,
    parent_path: str,
    population: str,
    config: dict | None,
) -> list[dict]:
    """LLM이 indicator와 의미적으로 매칭되는 row들을 식별해 반환.

    Args:
        rows: pop pre-filter 통과한 후보 row list.
        indicator: schema.indicator.
        claim_text: claim 원문 (앞 400자).
        parent_path: schema.parent_path.
        population: schema.population.
        config: 전체 config dict.

    Returns:
        매칭 row list. 빈 list면 LLM이 "이 표에 indicator 없다"고 판단.
    """
    if not rows or not indicator:
        return []

    # 1) unique 분류 값 추출 (ITM_NM/C1~C4_NM)
    field_values: dict[str, list[str]] = {}
    for f in _INDICATOR_FIELDS:
        vals = _extract_unique_values(rows, f, max_n=80)
        if vals:
            field_values[f] = vals

    if not field_values:
        logger.info("[row_matcher] 분류 컬럼 값 없음 → LLM 호출 skip")
        return []

    # 2) LLM 호출
    _rg_cfg = ((config or {}).get("kosis") or {}).get("relevance_guard") or {}
    model_tier = str(_rg_cfg.get("row_match_model_tier") or _rg_cfg.get("model_tier") or "light").strip().lower()

    prompt = _build_prompt(indicator, claim_text, parent_path, population, field_values)
    from structverify.utils.llm_client import LLMClient
    llm = LLMClient(config=(config or {}).get("llm") or {})
    try:
        raw = await llm.generate(
            prompt=prompt,
            system_prompt="KOSIS row 매칭 reviewer. JSON만 응답.",
            model_tier=model_tier,
        )
    except Exception as e:
        logger.warning(f"[row_matcher] LLM 호출 실패: {e}")
        return []

    matches, reasoning = _parse(raw)
    if not matches:
        logger.info(
            f"[row_matcher] LLM: 매칭 없음 — indicator={indicator!r} "
            f"reason={reasoning[:120]!r}"
        )
        return []

    # field/value swap 휴리스틱: LLM이 field 자리에 실제 매칭 값을 넣고
    # value 자리에 DT 같은 측정치(숫자/짧은 문자열)를 넣어 응답하는 케이스 회복.
    # (예: field='체외충격파쇄석기', value='18' → field='ITM_NM', value='체외충격파쇄석기')
    swapped: list[dict] = []
    for m in matches:
        f, v = m["field"], m["value"]
        if f not in _INDICATOR_FIELDS and (v.isdigit() or len(v) <= 3):
            logger.warning(
                f"[row_matcher] LLM field/value swap 감지 — "
                f"field={f!r} value={v!r} → value={f!r}로 재해석"
            )
            swapped.append({"field": _INDICATOR_FIELDS[0], "value": f})
        else:
            swapped.append(m)
    matches = swapped

    logger.info(
        f"[row_matcher] LLM 매칭 {len(matches)}건: {matches} "
        f"reason={reasoning[:80]!r}"
    )

    # 3) matches dict({field, value})로 rows 필터링
    def _norm(s: str) -> str:
        if not s:
            return ""
        s = str(s)
        for ch in (" ", "·", "ㆍ", "・", "/", "-", "_", "(", ")", "[", "]"):
            s = s.replace(ch, "")
        return s.strip().lower()

    norm_matches = [
        {"field": m["field"], "value_norm": _norm(m["value"]), "value_raw": m["value"]}
        for m in matches
    ]
    filtered: list[dict] = []
    field_mismatches: list[tuple[str, str]] = []  # (llm_field, actual_field) 로깅용

    for r in rows:
        matched = False
        for nm in norm_matches:
            target_v_norm = nm["value_norm"]
            if not target_v_norm:
                continue
            # 1차: LLM이 지정한 field 우선 검사
            v = r.get(nm["field"])
            if v is not None:
                v_norm = _norm(str(v))
                if target_v_norm in v_norm or v_norm in target_v_norm:
                    matched = True
                    break
            # 2차: LLM이 field를 잘못 지정한 경우 대비 — 모든 INDICATOR_FIELDS에서 value 검색.
            # LLM은 "체외충격파쇄석기"가 ITM_NM이라 우길 수 있는데 실제로는 C2_NM 같은 다른
            # 컬럼에 있음. value가 unique 분류라 다른 컬럼에 우연히 같은 단어 들어있을 가능성 낮음.
            for fname in _INDICATOR_FIELDS:
                if fname == nm["field"]:
                    continue  # 1차에서 이미 봄
                v = r.get(fname)
                if v is None:
                    continue
                v_norm = _norm(str(v))
                if target_v_norm in v_norm or v_norm in target_v_norm:
                    matched = True
                    field_mismatches.append((nm["field"], fname))
                    break
            if matched:
                break
        if matched:
            filtered.append(r)

    if field_mismatches:
        # LLM이 잘못 지정한 field → 실제 매칭된 field 로깅 (1회만)
        _ex = field_mismatches[0]
        logger.info(
            f"[row_matcher] LLM이 field={_ex[0]!r}로 지정했으나 실제로는 "
            f"{_ex[1]!r}에 매칭됨 — all-field fallback으로 회복 "
            f"({len(field_mismatches)} rows)"
        )

    logger.info(
        f"[row_matcher] LLM 매칭 적용: {len(rows)} rows → {len(filtered)} rows"
    )
    return filtered
