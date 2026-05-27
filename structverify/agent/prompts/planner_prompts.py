"""Plan Agent 프롬프트 (Phase C).

핵심 설계 원칙:
  1. **claim type 자동 분류** — LLM이 absolute / growth_rate / diff / ratio_comparison 판단
  2. **N data point 추출** — claim type에 따라 1-3개 데이터 점
  3. **시점 명확화** — "올 4월"/"지난해 같은 달" → 구체적 YYYY-MM
  4. **fallback 전략** — 1차 검색 실패 시 대안 키워드/접근법

LLM 응답 형식: 단일 JSON. 다른 텍스트 없이.
"""
from __future__ import annotations

from typing import Any


# ── Plan Prompt 본체 ──────────────────────────────────────────────────

PLAN_PROMPT_TEMPLATE = """당신은 한국 통계 팩트체크 시스템의 *Plan Agent*입니다.
주어진 *claim (주장)*을 검증하기 위해 *어떤 데이터가 필요한지* 계획을 세우세요.

## Claim 정보

원문 문장: {claim_text}
{schema_block}

{source_context_block}

## 임무

이 claim을 검증하기 위해:
1. claim의 *유형*을 분류하세요 (absolute / difference / growth_rate / comparison / ranking / unknown).
2. 검증에 *필요한 데이터 점*을 결정하세요.
3. 1차 시도가 실패할 경우의 *fallback 전략*을 세우세요.

### Claim 유형별 데이터 점 패턴

★ 분류 우선순위 — schema가 단일 value(prev_value 없음)이면 **absolute**.
   claim_text에 비교 문맥("~보다 적다", "~에 비해")이 있어도 *이 sub-claim*은
   자기 단일 값만 검증하면 됨. 비교 자체는 별도 layer에서 sub-claim verdict들을
   모아 도출. 단일 값 sub-claim을 comparison으로 잘못 분류하면 검증 시퀀스가
   망가져 calculate가 잘못 호출됨.

**absolute** — 단일 값 검증
  → 데이터 점 1개 (role=current)
  → calculation_formula 없음
  → 시퀀스: catalog_search → fetch_evidence → finish

**growth_rate** — 증가율/감소율 (schema.prev_value 있고 claim에 "%·증가율" 같은 비율 표현)
  → 데이터 점 2개 (current + prev)
  → calculation_formula: "(current - prev) / prev * 100"
  → 시퀀스: catalog_search → fetch_evidence(prev) → fetch_evidence(current) → calculate → finish

**difference** — 차이/변화량 (schema.prev_value 있고 차이가 *절대 단위*로 표현)
  → 데이터 점 2개 (current + prev)
  → calculation_formula: "current - prev"
  → 시퀀스: catalog_search → fetch_evidence(prev) → fetch_evidence(current) → calculate → finish

**comparison** — 두 *시점의 직접 값* 둘 다 명시 (예: "73.6% → 70.3%")
  → schema에 current+prev 두 값이 *모두 명시*된 경우만 해당
  → 데이터 점 2개 — 각각 *독립 매칭*
  → calculation_formula 없음 (calculate 호출하지 *말 것* — 비교는 *부등호*이지 수식이 아님)
  → 시퀀스: catalog_search → fetch_evidence × 2 → finish

**ranking** — 순위 (예: "1위", "하락폭이 가장 컸다")
  → 데이터 점 여러 개
  → calculate 호출 *금지*
  → 시퀀스: catalog_search → fetch_evidence × N → finish

**unknown** — 분류 불가. 데이터 점은 *최소한*만.
  → 시퀀스: catalog_search → fetch_evidence → finish

★ calculate 액션은 **growth_rate/difference에만** 사용. Python eval로 수식을
   계산하는 도구라 *부등호 비교에 쓸 수 없음*. absolute/comparison/ranking에서
   calculate를 시퀀스에 넣지 *말 것*.

★ **initial_steps에 반드시 finish step을 포함**할 것. plan이 finish로 끝나야
   검증이 자동 종료됨. finish 안 박으면 loop이 자율 결정으로 잘못된 액션을
   추가할 위험.

### 시점(time) 표기 규칙

- *YYYY-MM* 형식 권장 (예: "2025-04")
- 연간 데이터면 *YYYY* (예: "2025")
- *상대 표현 금지*: "지난해", "전년", "올해" 등을 그대로 쓰지 말고 *anchor_year를 보고* 구체적 연-월로 변환.
- anchor_year={anchor_year}, 추출된 schema.time_period={schema_time}

### Fallback 전략

1차 검색이 실패하기 쉬운 경우:
- *월별 데이터가 없는 표*만 매칭됨 → `alternative_keywords`에 "월별" 추가
- *합계출산율*처럼 일부 표에 *시계열만* 있음 → 다른 표명 시도

## 출력 형식

**반드시 아래 JSON만** 출력하세요. 다른 설명, 주석, 코드 펜스 절대 없음.

```json
{{
  "claim_type": "absolute | difference | growth_rate | comparison | ranking | unknown",
  "required_data": [
    {{
      "indicator": "출생아 수",
      "time": "2025-04",
      "population": "전체",
      "unit_hint": "명",
      "role": "current"
    }}
  ],
  "calculation_formula": "(current - prev) / prev * 100",
  "expected_result": 8.7,
  "expected_unit": "%",
  "verdict_logic": "calculation 결과가 expected_result와 ±5% 안에 있으면 match",
  "initial_steps": [
    {{
      "action": "catalog_search",
      "input": {{"query": "<검색 키워드>", "category": ["<분류>"], "top_k": 5}},
      "rationale": "데이터 소스에서 적합한 표 찾기"
    }},
    {{
      "action": "fetch_evidence",
      "input": {{"candidate_id": "<catalog_search 결과의 top id>", "params": {{}}}},
      "rationale": "후보 표에서 데이터 가져와서 row 매칭"
    }},
    {{
      "action": "finish",
      "input": {{}},
      "rationale": "검증 종료"
    }}
  ],
  "fallback": {{
    "use_original_text": false,
    "alternative_keywords": ["월별 인구동향", "출생사망혼인이혼"],
    "give_up_after_attempts": 5
  }},
  "notes": "이 claim은 ... (debugging용 자유 메모)"
}}
```

### role 값 (data point당)
- `current`: 비교의 *현재 시점* 값 (또는 단일 시점 값)
- `prev`: 비교의 *기준* 값 (이전 시점)
- `other`: 위에 안 맞는 경우 (ranking의 비교 대상들 등)

### action 값 (initial_steps의 단계)
- `catalog_search`: 데이터 소스 표 검색
- `fetch_evidence`: 후보 ID로 실제 수치 조회
- `read_original`: 원문 기사 일부 다시 읽기
- `calculate`: 모은 값으로 *수식* 계산 (growth_rate/difference 전용 — Python eval).
              *비교는 부등호이지 수식이 아니므로 calculate 호출 금지*.
- `finish`: 검증 종료 + verdict 결정. 모든 plan의 *마지막 step*으로 반드시 포함.

이제 위 형식대로 *JSON 한 개*만 출력하세요.
"""


def build_plan_prompt(
    claim_text: str,
    schema_info: dict[str, Any] | None = None,
    source_excerpt: str | None = None,
    anchor_year: str | int | None = None,
) -> str:
    """Plan prompt 인스턴스 생성.

    Args:
        claim_text: claim의 원문 문장. 필수.
        schema_info: schema_inductor 결과 (있으면). dict 키: indicator, value, unit, time_period, population, prev_value, prev_time_period, ...
        source_excerpt: 원문 기사의 일부 (있으면 — 맥락 보강).
        anchor_year: 문서의 anchor_year (있으면 — 시점 해소용).

    Returns:
        LLM에 그대로 전달할 prompt 문자열.
    """
    # schema block 구성
    if schema_info:
        lines = ["추출된 schema 정보 (이 sub-claim 전용 — claim_text 전체 해석보다 *이쪽을* 우선):"]
        # value_role을 최상단에 노출 — schema_inductor가 분기한 *역할*을 명시
        _role = schema_info.get("value_role")
        if _role:
            _role_to_plantype = {
                "base": "absolute (단일 값 검증, calculate 호출 금지)",
                "derived_rate": "growth_rate (비율 직접 계산)",
                "derived_difference": "difference (절대 차이 계산)",
            }
            mapping_hint = _role_to_plantype.get(_role, _role)
            lines.append(
                f"  ★ value_role: {_role!r} → claim_type을 *{mapping_hint}*로 설정할 것. "
                f"같은 문장의 다른 sub-claim과 헷갈리지 마세요."
            )
        for k in ("indicator", "value", "unit", "time_period", "population",
                  "prev_value", "prev_time_period", "prev_phrase",
                  "is_approximate", "modifier", "parent_path"):
            v = schema_info.get(k)
            if v is not None and v != "":
                lines.append(f"  - {k}: {v!r}")
        schema_block = "\n".join(lines)
        schema_time = schema_info.get("time_period") or "(없음)"
    else:
        schema_block = "추출된 schema 정보: (없음 — claim 원문에서만 추론)"
        schema_time = "(없음)"

    # source context block (옵션)
    if source_excerpt:
        # 너무 길면 자름 (1000자)
        excerpt = source_excerpt.strip()
        if len(excerpt) > 1000:
            excerpt = excerpt[:1000] + "...[잘림]"
        source_context_block = f"## 원문 기사 일부 (맥락)\n\n{excerpt}\n"
    else:
        source_context_block = ""

    anchor_year_str = str(anchor_year) if anchor_year else "(없음)"

    return PLAN_PROMPT_TEMPLATE.format(
        claim_text=claim_text,
        schema_block=schema_block,
        source_context_block=source_context_block,
        anchor_year=anchor_year_str,
        schema_time=schema_time,
    )
