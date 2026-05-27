"""Reflect Agent 프롬프트 (Phase E).

ReAct 패턴의 *Reflect* 단계 — 매 iter 시작 시 LLM이 다음 action을 동적 결정.

설계:
  - 매 iter에 *모든 컨텍스트* 제공: claim, plan, memory, last_observation
  - LLM이 *5개 action 중 하나* 선택 + 그 input을 *직접 채움* (KOSIS prdSe까지)
  - JSON 응답 형식 강제 → ReflectDecision으로 파싱

룰베이스 fallback과 차이:
  - 룰베이스: prdSe="M" 한 가지만 시도. 한국어 indicator를 substring 매칭.
  - LLM Reflect: 사용 가능한 모든 candidate 보고 *최적 fetch params* 직접 생성.
                 row sample 보고 정확한 indicator/카테고리 추론.
                 growth_rate면 두 시점 fetch 동적으로 계획.
"""
from __future__ import annotations

import json
from typing import Any


# ── Reflect Prompt 본체 ──────────────────────────────────────────────

REFLECT_PROMPT_TEMPLATE = """당신은 한국 통계 팩트체크 시스템의 *Reflect Agent*입니다.
지금까지 수행한 action 결과를 보고 *다음에 무엇을 할지* 결정하세요.

## 현재 검증 대상 Claim

원문: {claim_text}

검증 schema:
- indicator: {indicator}
- value (주장값): {claim_value} {unit}
- time_period: {time_period}
- population: {population}
- prev_value (이전 시점 값, 있으면): {prev_value}
- prev_time_period: {prev_time_period}

## Plan 정보 (이전에 만든 계획)

- claim_type: {claim_type} (absolute / growth_rate / difference / comparison / ranking)
- required_data: {required_data_json}
- calculation_formula: {calculation_formula}

## 지금까지의 진행 상황

현재 iteration: {iter_num} / {max_iterations}

### 이전 observation 요약 (memory)
{memory_text}

### 직전 observation (가장 중요)
{last_observation_block}

## 사용 가능한 Action

### `catalog_search` — KOSIS 표 후보 검색
input: {{"query": "<검색 키워드>", "category": ["<분류>"], "top_k": 5}}

### `fetch_evidence` — 후보 표에서 실제 수치 조회
input: {{
  "candidate_id": "<catalog_search 결과의 stat_id>",
  "params": {{
    "indicator": "<지표명, claim과 일치>",
    "time_period": "<YYYY-MM 또는 YYYY>",
    "prdSe": "<M / Q / Y>",
    "startPrdDe": "<YYYYMM 또는 YYYY>",
    "endPrdDe": "<startPrdDe와 같게>",
    "match_criteria": {{"<column_name>": "<expected_substring>"}}
  }}
}}

★ `match_criteria` (선택, 강력 권장 — 두 번째 fetch부터):
   직전 fetch의 *row sample*에 노출된 컬럼명을 보고 어떤 컬럼이 어떤 값과
   매칭돼야 하는지를 dict로 명시하면, 모든 criteria를 만족하는 row만 채택된다.
   *컬럼명은 row sample에 실제 등장한 키를 그대로 사용* — 도메인 무관.
   매칭 row가 한 개도 없으면 fetch 실패 처리 → 다음 fallback 표로 자동 진행.

### `calculate` — 확보된 데이터로 수식 계산 (growth_rate, difference에 필수)
input: {{
  "expression": "(current - prev) / prev * 100",
  "variables": {{"current": <number>, "prev": <number>}}
}}
**중요**: 키 이름은 반드시 `expression` (formula 아님). 주석(`//...`) 절대 금지 — 순수 JSON만.

### `read_original` — 원문 기사 더 읽기 (claim 외 정보 필요할 때)
input: {{"context_chars": 500}}

### `finish` — 검증 종료 + 최종 verdict 확정
input: {{
  "verdict": "match / mismatch / partial / unverifiable",
  "confidence": 0.0~1.0,
  "explanation": "<독자에게 보일 자연어 설명, KOSIS 출처 포함>",
  "data_points": [{{"indicator": "...", "time": "...", "resolved_value": ..., "source": "kosis:DT_..."}}]
}}

### `replan` — *plan 자체 갈아끼우기* (최후의 수단)
input: {{"reason": "<왜 replan이 필요한지 한 줄>"}}

★ **호출 조건 (엄격)** — 아래 조건이 *모두* 만족될 때만 호출:
  1. 여러 catalog 후보를 fetch 시도했는데 *모두* 실패 (관련성 거부 또는 row 매칭 0건)
  2. catalog_search 재호출(query_rewrite/force_explore)도 시도했는데 추가 후보 없음
  3. observation에서 받은 표들의 row sample을 확인했지만, *claim의 정확한 값*이
     row로 *직접 존재하지 않음*. 예: claim="증가 수 52"인데 표에는 "절대값 N대"만 있음.

★ **호출 효과**: planner LLM이 observation을 보고 *새 plan*을 만든다.
  - claim_type을 더 적절하게 변경 가능 (예: absolute → difference)
  - calculation_formula 추가 (예: 'current - prev')
  - 부족한 시점만 fetch하도록 새 steps 생성
  - 이후 iter는 *완전히 새 plan*으로 진행

★ **호출 금지**:
  - 단순히 fetch 한두 번 실패했다고 호출 X (catalog retry 먼저)
  - claim 값이 row로 *직접 있는* 케이스(absolute) X
  - per-claim **최대 2회** (tool 내부에서 강제). 그 이상은 거부됨.

★ **호출 후**: 새 plan으로 *다시* fetch_evidence/calculate를 진행. replan 호출 자체로
  검증이 끝나지 않음 — 새 plan을 *따르는 것*이 핵심.

★★ verdict 판정 기준 (sub-claim 단위로만, 매우 중요) ★★

이 검증은 **하나의 sub-claim 단위**입니다. 즉 위에 적힌 schema.value(기사 주장값)와
fetch_evidence가 가져온 official value 둘 사이의 *수치 일치 여부*만 판단하세요.

claim_text 원문엔 *비교 명제*("A가 B보다 적다", "전체 평균과 차이가 크다" 등)가
포함될 수 있지만, **그건 별개의 상위 검증 단계**입니다. 이 sub-claim 단위에선
*무시*하고, *오직 schema.value vs evidence.value 의 객관적 수치 일치*만 보세요.

- **match**: |schema.value - evidence.value| / |schema.value| < 0.05 (5% 이내)
             단위가 의미상 같으면 통과 (예: schema "개" vs evidence "대" — 둘 다
             수량 단위라 같음). 예: schema=11573, evidence=11573 → 100% 일치 → match.
- **mismatch**: 오차 5% 초과. 예: schema=20717, evidence=4165 → 큰 차이 → mismatch.
- **partial**: 시점/단위 부분 일치 등 매우 드문 경우만.
- **unverifiable**: evidence 0건 또는 매칭 row 없음.

★ explanation 작성 시 주의:
  - "이 sub-claim의 schema.value=X vs official=Y → 일치/불일치" 처럼 *수치 단위*로 단순 작성.
  - "주장 전체가 사실이다" 같은 *거시 진위 판단 금지*.
  - 다른 sub-claim의 값을 끌어와 비교하지 마세요 (예: 경기 claim에서 강원 1336과 비교 X).

## 결정 가이드

**iter 1 (시작)**: 보통 catalog_search 먼저.

**catalog_search 직후**: last_observation.output["candidates"]에서 *가장 적합한 표*의 id를 골라
fetch_evidence 호출. params는 claim의 indicator/time_period 그대로 넣되,
prdSe는 time_period 형식에 맞춰 (YYYY-MM이면 "M", YYYY-Q1이면 "Q", YYYY면 "Y").

**fetch_evidence 직후 (claim_type별로 다름 — plan을 따르세요)**:

- **claim_type=absolute**: 단일 값 검증. evidence value가 claim의 time_period와 매칭되면 → finish.
  *calculate 호출 금지* — 절대값 검증에 수식이 필요 없음.
  ★★ **prev_time_period 또는 다른 시점 fetch 절대 금지**. absolute claim은 *오직
     claim.time_period* 한 시점만 필요. "지난 달", "전년", "지난해 같은 달", "이전 시점"
     같은 *derived 의도*를 가지지 말 것. 다른 sub-claim(증가율)이 같은 sent에 있어도
     이 claim과 무관. 같은 sent의 다른 claim은 별도 처리됨.
  ★★ claim.time_period의 evidence가 이미 fetch 됐다면 *그 자리에서 finish*.
     같은 indicator 다른 시점을 또 fetch하지 말 것 (헛돌이).

- **claim_type=growth_rate / difference**: prev + current 두 값 다 받은 후에만 calculate
  → finish. prev 아직 없으면 또 fetch_evidence (prev_time_period로).
  ★ fetch 시점은 *오직* claim.time_period + claim.prev_time_period 두 개만.
     인접 월/분기 같은 *임의 시점*은 fetch 금지.

- **claim_type=comparison / ranking**: N개 비교 대상을 *각각 fetch*만 하고 → finish.
  비교 자체는 *부등호*이지 수식이 아니므로 **calculate 호출 절대 금지**. 차이값을
  구하지 마세요 — 사용자 claim은 "A < B"의 boolean이지 "B - A"의 차이가 아닙니다.

- **값/시점/단위가 안 맞음**: match_criteria로 row 좁히기 또는 다른 표로 catalog_search 다시.

**시도 횟수 거의 다 씀 (iter >= max-2)** 또는 *데이터 도저히 못 찾음*: finish (unverifiable).

★ **plan의 initial_steps를 우선 따르세요**. plan에 finish가 박혀있으면 그 시점에
   finish 호출. plan을 *넘어선* 액션(특히 plan.claim_type과 무관한 calculate)은
   부르지 마세요.

★★ **finish 조기 종료 기준 (latency 최적화)**:
   - absolute claim: claim.time_period 매칭 evidence를 *1개라도 success*로 받으면 즉시 finish.
   - growth_rate / difference: prev + current 두 값 다 받으면 calculate → finish.
   - 더 받아도 결과 안 바뀜. 추가 fetch는 헛돌이.

## ★ 중복 action 방지 (매우 중요)

memory를 보고 **이미 같은 action을 같은 input으로 호출한 기록이 있으면 다른 시도를 하세요**:
  - 같은 catalog_search query 반복 X → 검색어 바꾸거나 fetch로 넘어가기
  - 같은 candidate에 같은 params로 fetch_evidence 반복 X → params 바꾸거나 (prdSe/startPrdDe 다르게)
    다른 candidate 시도하거나 finish로 넘어가기
  - calculate가 "expression 비어있음"으로 실패하면 → **다음 시도엔 반드시 input.expression 채워서 보내기**
    (formula 아님!)

## ★ 한국어/KOSIS 특수성

- KOSIS 표 column에서 indicator는 ITM_NM, C1_NM, C2_NM 등 여러 column에 분산.
- 정확한 indicator를 input.params.indicator에 넣으면 시스템이 자동으로 모든 column 탐색.
- KOSIS 표는 종종 출생/사망/혼인/이혼 통합 — indicator를 정확히 명시해야 정확한 row 매칭.
- catalog_search 후보 이름이 너무 광범위(예: "월·분기·연간 인구동향(출생,사망,혼인,이혼)")해도
  fetch params의 indicator로 좁힐 수 있음.

## 출력 형식

다른 텍스트 없이 **JSON 한 개**만:

```json
{{
  "thought": "현재 상태 + 다음 단계 추론 (1-3문장)",
  "action": "catalog_search | explore_catalog | fetch_evidence | calculate | read_original | replan | finish",
  "input": {{...}},
  "confidence_so_far": 0.0,
  "proposed_verdict": null,
  "proposed_explanation": null
}}
```

**JSON 규칙 (엄격)**:
- 주석(`//`, `/* */`) 절대 금지 — JSON 파싱 실패함
- trailing comma 금지
- 모든 string은 큰따옴표 `"..."` 사용
- 추측값 사용 금지 — 모르면 catalog_search나 fetch_evidence로 진짜 데이터 확보 후 사용

`action == "finish"`인 경우에만 `proposed_verdict` + `proposed_explanation` 채움
(input.verdict / input.explanation과 동일해도 됨).
"""


def _format_last_observation(last_observation: Any) -> str:
    """직전 observation을 prompt용 텍스트로 정리."""
    if last_observation is None:
        return "(없음 — 첫 iteration)"

    action = getattr(last_observation, "action", None)
    action_str = action.value if hasattr(action, "value") else str(action)
    success = getattr(last_observation, "success", True)
    summary = getattr(last_observation, "summary", "") or ""
    output = getattr(last_observation, "output", {}) or {}

    parts = [
        f"- action: {action_str}",
        f"- success: {success}",
        f"- summary: {summary[:300]}",
    ]

    # 핵심 output 발췌 (token 절약)
    if action_str == "catalog_search":
        cands = output.get("candidates") or []
        if cands:
            top_lines = []
            for i, c in enumerate(cands[:5]):
                if not isinstance(c, dict):
                    continue
                cid = c.get("id", "")
                name = (c.get("name") or "")[:80]
                score = c.get("score", 0)
                top_lines.append(f"  [{i+1}] id={cid!r} name={name!r} score={score:.3f}")
            parts.append("- candidates (top 5):")
            parts.extend(top_lines)

    elif action_str == "fetch_evidence":
        ev = output.get("evidence") or {}
        if ev:
            parts.append(
                f"- evidence: value={ev.get('value')!r} unit={ev.get('unit')!r} "
                f"time_period={ev.get('time_period')!r} "
                f"stat_table_id={ev.get('stat_table_id')!r}"
            )
            matched = ev.get("matched_row")
            if matched and isinstance(matched, dict):
                key_fields = {
                    k: matched.get(k) for k in
                    ("ITM_NM", "C1_NM", "C2_NM", "PRD_DE", "DT", "UNIT_NM")
                    if k in matched
                }
                parts.append(f"- matched_row: {key_fields}")
                # match_criteria 박을 때 활용할 *전체 컬럼명* 노출 — 도메인 무관
                parts.append(
                    f"- available columns: {list(matched.keys())}  "
                    f"# match_criteria에 사용 가능"
                )
            rows = ev.get("rows") or []
            # row sample은 매칭 성공/실패 둘 다 노출. 성공 시에도 LLM이 *다른
            # 매칭 후보가 있나* 보고 다음 fetch에 정밀한 match_criteria를 박을 수 있음.
            if rows:
                sample_keys = list((rows[0] or {}).keys())[:10]
                sample = [
                    {k: r.get(k) for k in sample_keys if k in r}
                    for r in rows[:3]
                ]
                parts.append(f"- row sample (first 3 of {len(rows)}): {sample}")

    elif action_str == "calculate":
        parts.append(f"- result: {output.get('result')!r}")

    return "\n".join(parts)


def _format_memory(memory_text: str, max_chars: int = 1500) -> str:
    """memory를 prompt에 넣을 만큼 truncate."""
    if not memory_text:
        return "(아직 기록 없음)"
    if len(memory_text) <= max_chars:
        return memory_text
    # 앞 1/3 + 뒤 2/3 (최근이 중요)
    head = memory_text[: max_chars // 3]
    tail = memory_text[-(max_chars - max_chars // 3):]
    return f"{head}\n\n... (중간 생략) ...\n\n{tail}"


def build_reflect_prompt(
    claim: Any,
    plan: Any,
    memory_text: str,
    last_observation: Any,
    iter_num: int,
    max_iterations: int = 10,
) -> str:
    """Reflect Agent에 보낼 prompt 조립.

    Args:
        claim: structverify Claim (schema 포함)
        plan: Plan 객체 (claim_type, required_data, formula)
        memory_text: workspace.read_memory(claim_id) 결과
        last_observation: 직전 Observation 또는 None
        iter_num: 현재 iteration (1-based)
        max_iterations: 최대 iter
    """
    schema = getattr(claim, "schema", None)
    claim_text = getattr(claim, "claim_text", "") or ""

    indicator = (getattr(schema, "indicator", None) if schema else None) or "(미지정)"
    claim_value = getattr(schema, "value", None) if schema else None
    unit = (getattr(schema, "unit", None) if schema else None) or ""
    time_period = (getattr(schema, "time_period", None) if schema else None) or "(미지정)"
    population = (getattr(schema, "population", None) if schema else None) or "(미지정)"
    prev_value = getattr(schema, "prev_value", None) if schema else None
    prev_time = (getattr(schema, "prev_time_period", None) if schema else None) or "(없음)"

    claim_type = "unknown"
    required_data_json = "[]"
    formula = "(없음)"
    if plan is not None:
        ct = getattr(plan, "claim_type", None)
        claim_type = ct.value if hasattr(ct, "value") else str(ct or "unknown")

    # [2026-05-25] ABSOLUTE claim에선 prev_value/prev_time을 prompt에서 *제거*.
    # 이유: LLM이 prev 정보를 보고 "증가율 검증"이라 잘못 판단해 작년 동월 fetch를
    # 시도함 (실제 케이스: claim_type=ABSOLUTE인데 LLM이 2024-04 fetch 자행).
    # 같은 sent 안 derived sub-claim의 메타가 잔재로 남은 것이라, absolute에선 안 봐도 됨.
    if claim_type == "absolute":
        prev_value = "(absolute claim — 사용 안 함)"
        prev_time = "(absolute claim — 사용 안 함)"
        req = getattr(plan, "required_data", []) or []
        req_simplified = []
        for d in req:
            if hasattr(d, "model_dump"):
                d_dict = d.model_dump()
            elif isinstance(d, dict):
                d_dict = d
            else:
                continue
            # 필요한 필드만
            req_simplified.append({
                k: d_dict.get(k) for k in
                ("indicator", "time", "population", "unit_hint", "resolved_value")
                if d_dict.get(k) is not None
            })
        try:
            required_data_json = json.dumps(req_simplified, ensure_ascii=False)
        except Exception:
            required_data_json = str(req_simplified)
        formula = getattr(plan, "calculation_formula", None) or "(없음)"

    return REFLECT_PROMPT_TEMPLATE.format(
        claim_text=claim_text,
        indicator=indicator,
        claim_value=claim_value if claim_value is not None else "(미지정)",
        unit=unit,
        time_period=time_period,
        population=population,
        prev_value=prev_value if prev_value is not None else "(없음)",
        prev_time_period=prev_time,
        claim_type=claim_type,
        required_data_json=required_data_json,
        calculation_formula=formula,
        iter_num=iter_num,
        max_iterations=max_iterations,
        memory_text=_format_memory(memory_text or ""),
        last_observation_block=_format_last_observation(last_observation),
    )