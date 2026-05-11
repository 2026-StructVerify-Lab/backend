"""
detection/schema_inductor.py — Dynamic Schema Induction (Step 5)

[김예슬 - 2026-04-22]
- SCHEMA_INDUCTION_PROMPT: 도메인 컨텍스트 주입 + 예시 강화
- _safe_float(): 다양한 수치 표현 파싱 ("64.2%", "약 64" 등)
- _validate_schema(): 최소 유효성 검증
- 재시도 로직 추가 (최대 2회)

[김예슬 - 2026-04-24]
- generate_json() → generate_structured() 으로 교체
  · 기존: LLM이 JSON 텍스트 생성 → 직접 파싱 (실패 가능, 재시도 필요)
  · 변경: Structured Outputs (HCX-007) → JSON Schema 보장 (파싱 실패 없음)
- CLAIM_SCHEMA_JSON_SCHEMA: ClaimSchema에 대응하는 JSON Schema 정의 추가
- OpenAI fallback도 response_format으로 동일하게 처리
- 재시도 로직 제거 (Structured Outputs는 파싱 실패 자체가 없음)

[참고] CLOVA Studio Structured Outputs
  https://api.ncloud-docs.com/docs/en/clovastudio-chatcompletionsv3-so
[참고] AutoSchemaKG (arXiv 2505.23628)
"""
from __future__ import annotations

import re
from typing import Any

from structverify.core.schemas import (
    Claim, ClaimSchema, EvidencePlan, EvidenceRequirement, ValueRole,
)
from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

# 도메인별 indicator 힌트
DOMAIN_HINTS: dict[str, str] = {
    "agriculture": "농가 수, 경작면적, 수확량, 고령화비율, 후계농 비율, 농업소득 등",
    "economy":     "경제성장률, 소비자물가지수, 수출액, 취업자 수, 산업생산지수 등",
    "finance":     "금리, 환율, 주가지수, 대출잔액, 가계부채비율 등",
    "population":  "인구수, 합계출산율, 기대수명, 고령화비율, 인구증가율 등",
    "employment":  "고용률, 실업률, 임금상승률, 취업자 수, 근로시간 등",
    "healthcare":  "의료기관 수, 사망률, 질환자 수, 의료비, 건강보험료 등",
    "education":   "학생 수, 진학률, 교육비, 학교 수, 졸업률 등",
}

# ── Structured Outputs용 JSON Schema 정의 ────────────────────────────────
# HCX-007이 반드시 이 형식으로 반환하도록 강제
CLAIM_SCHEMA_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "indicator": {
            "type": "string",
            "description": "측정하는 핵심 지표 (예: 65세 이상 경영주 비율)"
        },
        "time_period": {
            "type": "string",
            "description": "기준 연도/시점 (예: 2023, 2023년 1분기)"
        },
        "unit": {
            "type": "string",
            "description": "수치 단위 (예: %, 만명, ha, 억원)"
        },
        "population": {
            "type": "string",
            "description": "대상 집단/범위 (예: 과수 농가, 전국, 15~64세)"
        },
        "value": {
            "type": "number",
            "description": (
                "주장에 나온 수치. 원문 그대로의 숫자 (단위 변환 금지). "
                "예: '6.7% 증가' → 6.7 (NOT 670, NOT 0.067). "
                "예: '14.8도' → 14.8. "
                "예: '34년 만에' → 34. "
                "퍼센트 기호가 있어도 곱하거나 나누지 마세요. 그냥 숫자만."
            )
        },
        "value_role": {
            "type": "string",
            "enum": ["measurement", "threshold", "delta", "rank", "ratio", "none"],
            "description": (
                "value의 의미적 역할. measurement만 KOSIS와 직접 비교됨. "
                "threshold/rank/none은 비교 제외. delta/ratio는 별도 검증 필요."
            ),
        },
        "source_reference": {
            "type": "string",
            "description": "주장에 언급된 출처 기관/보고서"
        },
        "graph_schema_candidates": {
            "type": "array",
            "description": "Knowledge Graph 노드/엣지 후보",
            "items": {
                "type": "object",
                "properties": {
                    "node_type": {"type": "string"},
                    "label": {"type": "string"},
                    "edge_type": {"type": "string"},
                    "from": {"type": "string"},
                    "to": {"type": "string"}
                }
            },
            "maxItems": 6
        },
        "evidence_plan": {
            "type": "object",
            "description": (
                "검증에 필요한 시점 계획. "
                "value_role에 따라 형태가 달라짐: "
                "measurement → 1개(primary), delta/ratio → 2개(endpoint_a, endpoint_b), "
                "rank/threshold/none → 0개(KOSIS 직접 비교 불가)."
            ),
            "properties": {
                "combiner": {
                    "type": "string",
                    "enum": ["direct", "delta", "ratio_pct"],
                    "description": (
                        "evidences를 결합하는 방식. "
                        "direct=measurement 1개 그대로, "
                        "delta=endpoint_a − endpoint_b, "
                        "ratio_pct=(endpoint_a − endpoint_b)/endpoint_b × 100."
                    ),
                },
                "requirements": {
                    "type": "array",
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {
                                "type": "string",
                                "enum": ["primary", "endpoint_a", "endpoint_b"],
                            },
                            "label": {"type": "string"},
                            "indicator": {"type": "string"},
                            "time_period": {"type": "string"},
                            "population": {"type": "string"},
                        },
                        "required": ["role", "time_period"],
                    },
                },
            },
            "required": ["combiner", "requirements"],
        }
    },
    "required": ["indicator", "time_period", "unit", "population", "value_role", "evidence_plan"]
}

SCHEMA_INDUCTION_PROMPT = """아래 주장에서 공식 통계 검증에 필요한 핵심 정보를 추출하세요.

검증 대상 문장: "{claim_text}"
문맥 (참고용): {context}
도메인: {domain}
{domain_hint}
{temporal_hint}
{memory_hint}

[추출 기준]
- indicator: **수치가 측정하는 본체** (KOSIS 통계표명에 등장할 만한 *짧은 공식 용어*)
  · 문맥을 참고하여 "이는", "해당" 등 대명사 해소
  · ⚠️ **분류 축 단어를 indicator에 섞지 마세요**. 분류 축은 population으로 분리.
  · 분류 축 예시: "대졸이상", "청년", "여성", "20대", "외국인", "수도권" 등
    → 이런 단어는 *누구를/어디를* 측정하는지 (= population)이지 *무엇을* 측정하는지가 아님.
  · KOSIS 공식 용어 예시 (indicator 본체로 적절):
    · 고용: 쉬었음 인구, 비경제활동인구, 취업자수, 실업률, 임금근로자수, 고용률
    · 인구: 출생아수, 사망자수, 혼인건수, 이혼건수, 합계출산율, 1인 가구
    · 가구/소득: 가구소득, 가처분소득, 가계지출
    · 환경/기상: 연평균기온, 평균최저기온, 평균최고기온, 강수량

  [올바른 분리 예시]
  "대졸이상 청년 쉬었음 비율이 9.8%로"
    → indicator: "쉬었음 인구"  (또는 "쉬었음 비율")
    → population: "대졸이상 청년"
  "여성 65세 이상 1인 가구가 30%"
    → indicator: "1인 가구"
    → population: "여성 65세 이상"
  "외국인 근로자 평균 임금"
    → indicator: "임금"  (또는 "평균임금")
    → population: "외국인 근로자"
  "수도권 출생아수 6.7% 증가"
    → indicator: "출생아수"
    → population: "수도권"

  [잘못된 예시 — 절대 이렇게 박지 마세요]
  ❌ indicator: "대졸이상 청년 쉬었음 비율"  (분류 축 섞임)
  ❌ indicator: "여성 65세 이상 1인 가구"   (분류 축 섞임)
  ❌ indicator: "수도권 출생아수"            (분류 축 섞임)

- time_period: 기준 연도/시점
  · 위 [시점 정보]가 제공되면 그 절대 시점을 그대로 사용하세요 ("작년" 같은 표현 대신).
  · 제공되지 않은 경우만 본문에서 직접 추출.
- unit: 수치 단위
- population: 대상 집단/범위 — 위 indicator에서 빼낸 분류 축이 여기로
- value: **검증 대상 문장**에 직접 나온 수치만 추출 (문맥의 수치는 사용 금지, 없으면 null)
- value_role: 위 value의 의미적 역할 — 정확히 하나 선택
  · "measurement": 직접 측정값 (예: "평균기온 14.8도", "인구 5천만명", "출생아 2만 171명")
  · "threshold":   기준선/임계값 ("14도를 넘겼다", "1.5도 이상", "20도를 웃돌았다"의 20)
  · "delta":       변화량/차이 ("2.3도 웃돌았다"의 2.3, "0.18도 더 높다", "0.04명 증가했다"의 0.04)
  · "rank":        순위/기록 ("4위", "역대 1위", "34년 만에 최대 증가"의 34, "6년 만에 최고치"의 6)
                   ⚠️ "N년 만에 최대/최고/최저"의 N은 기록을 깬 시간 간격 — rank로 분류
  · "ratio":       비율/배수/% 증감 ("2.6배 늘었다", "30% 증가율", "6.7% 늘었다"의 6.7)
  · "none":        위 어느 것도 아님
- source_reference: 언급된 출처 (없으면 null)
- graph_schema_candidates: KG 노드/엣지 후보 (최대 6개)


[evidence_plan — 검증 계획 생성 (중요)]
이 주장을 KOSIS로 검증하려면 어떤 시점들의 측정값이 필요한가? value_role에 따라 결정:

· value_role=measurement → combiner="direct", requirements=[{{role:"primary", time_period:claim의 시점, ...}}]
  예: claim="2023년 평균기온 14.8도" → primary 1개

· value_role=ratio → combiner="ratio_pct", requirements 2개
  - endpoint_a (현재): claim의 시점, indicator는 비율의 기준이 되는 측정값
  - endpoint_b (기준): 비교 대상 시점, indicator 동일
  예: claim="올 4월 출생아 6.7% 증가" (anchor=2025)
    → endpoint_a: time="2025-04", indicator="출생아 수"
    → endpoint_b: time="2024-04", indicator="출생아 수"

· value_role=delta → combiner="delta", requirements 2개 (ratio와 동일 구조)
  예: claim="합계출산율 0.04명 증가"
    → endpoint_a: 현재 시점 합계출산율
    → endpoint_b: 비교 시점 합계출산율

· value_role=threshold/rank/none → combiner="direct", requirements=[]  (검증 안 함)

[중요]
- value는 반드시 "검증 대상 문장"에 있는 숫자여야 합니다. 문맥의 숫자 사용 금지.
- value_role 정확히 분류:
  · "14도를 넘었다"의 14는 threshold (measurement 아님)
  · "6.7% 늘었다"의 6.7은 ratio (measurement 아님)
  · "2.3도 웃돌았다"의 2.3은 delta
- evidence_plan의 indicator는 비율/차이의 *기준이 되는 measurement 지표명*을 쓰세요.
  "출생아 수 증가율" 6.7%의 evidence_plan indicator는 "출생아 수"입니다 (증가율 아님).
- requirements의 time_period는 절대 표현(YYYY 또는 YYYY-MM)으로."""


async def induce_schemas(
    claims: list[Claim],
    config: dict | None = None,
    graph: "ClaimGraph | None" = None,
    memory: "DocumentMemory | None" = None,   # [v6.4 추가] 옵셔널 — 기존 호출자 호환
) -> list[Claim]:
    """
    각 주장에서 ClaimSchema를 동적으로 유도한다.

    [v6 멀티홉] graph가 주어지면 claim의 시점을 멀티홉으로 resolve해서
    prompt에 절대 시점 hint로 주입. LLM은 굳이 "작년"을 다시 해석할 필요 없음.

    [v6.4 추가] memory가 주어지면 이전 처리한 claim들의 schema/plan을 prompt에
    "이전 claim 컨텍스트"로 주입. "이는", "같은", "전년" 등 지시어 해석에 사용.

    Structured Outputs(HCX-007) 사용으로 JSON 파싱 실패 없음.
    실패 시 재시도 없이 schema=None으로 처리.
    """
    config = config or {}
    llm = LLMClient(config=config.get("llm", {}))
    success, fail = 0, 0

    for claim in claims:
        domain = config.get("detected_domain", "general")
        domain_hint = f"주요 지표 예시: {DOMAIN_HINTS[domain]}" if domain in DOMAIN_HINTS else ""

        # [v6] context 텍스트 (변경 없음 — 보조 신호)
        context = getattr(claim, "context_text", None) or claim.claim_text

        # [v6 멀티홉] 그래프에서 시점 해소 결과를 hint로 주입
        # 다층 방어:
        #   1) claim 단위로 resolved 시점이 있으면 그것을 직접 hint
        #   2) 없어도 anchor_year는 항상 hint에 포함 (LLM이 본문 표현을 anchor로 풀게)
        temporal_hint = ""
        if graph is not None:
            prov = graph.temporal_provenance(claim)
            anchor_year = graph.get_anchor_year()

            if prov and prov.get("resolved"):
                # 1차: 그래프에서 직접 풀린 절대 시점
                temporal_hint = (
                    f"\n[시점 정보 — 그래프 해소 결과]\n"
                    f"- 원문 표현: {prov.get('expression')}\n"
                    f"- 해소된 절대 시점: {prov['resolved']}\n"
                    f"- 근거: {prov.get('basis') or '문서 anchor 기반'}\n"
                    f"위 절대 시점을 time_period로 사용하세요."
                )
            elif anchor_year is not None:
                # 2차: claim 단위 해소는 실패했지만 anchor_year는 있음
                # → LLM이 본문의 "작년/지난해/재작년" 등을 anchor 기준으로 풀게
                temporal_hint = (
                    f"\n[시점 정보 — 문서 anchor]\n"
                    f"- 이 문서의 기준 연도(anchor_year): {anchor_year}\n"
                    f"- 본문에 '작년/지난해/재작년/올해' 같은 상대 표현이 있으면\n"
                    f"  anchor_year를 기준으로 절대 연도(예: {anchor_year-1}, {anchor_year-2})로 풀어\n"
                    f"  time_period에 절대값으로 적으세요.\n"
                    f"- '산업화 이전' 같은 비-숫자 표현은 그대로 두어도 됩니다."
                )

        # [v6.4 추가] memory_hint — 이전 처리한 claim들의 schema/plan을 prompt에 주입
        # "이는", "같은 달", "전년" 같은 지시/대명사 해석에 사용
        memory_hint = ""
        if memory is not None:
            memory_hint = "\n" + memory.recent_context_for_prompt(max_items=4)

        schema = await _induce_single(
            llm, claim.claim_text, domain, domain_hint,
            context=context, temporal_hint=temporal_hint,
            memory_hint=memory_hint,   # [v6.4 추가]
        )
        if schema:
            claim.schema = schema
            success += 1

            # [v6.7 추가] evidence_plan.requirements의 빈 indicator를 schema.indicator로 자동 채움
            # LLM이 measurement 케이스에서 req.indicator를 가끔 None으로 두는 비일관성 발견.
            # schema.indicator는 이미 신뢰할 수 있는 값이므로 fallback으로 사용.
            # 이건 룰 매핑이 아니라 LLM이 빠뜨린 정보를 schema 본인 값으로 보강.
            if schema.evidence_plan and schema.evidence_plan.requirements:
                filled_count = 0
                for req in schema.evidence_plan.requirements:
                    if not req.indicator and schema.indicator:
                        req.indicator = schema.indicator
                        filled_count += 1
                if filled_count:
                    logger.debug(
                        f"  └─ [v6.7] requirements.indicator 보강: "
                        f"{filled_count}건 → '{schema.indicator}'"
                    )

            logger.info(
                f"스키마 유도: {claim.sent_id} "
                f"indicator={schema.indicator}, value={schema.value}, "
                f"time_period={schema.time_period}, "
                f"value_role={schema.value_role.value}"
            )
            # [v6.3] evidence_plan 로깅 — 검증에 필요한 시점들 확인
            if schema.evidence_plan and schema.evidence_plan.requirements:
                reqs = schema.evidence_plan.requirements
                req_str = ", ".join(
                    f"{r.role}@{r.time_period}(ind={r.indicator})"
                    for r in reqs
                )
                logger.info(
                    f"  └─ plan: combiner={schema.evidence_plan.combiner} "
                    f"requirements=[{req_str}]"
                )
            elif schema.value_role.value in ("ratio", "delta"):
                logger.warning(
                    f"  └─ [경고] value_role={schema.value_role.value}인데 "
                    f"evidence_plan.requirements 비어있음 — 검증 불가"
                )

            # [v6.4 추가] memory에 이 claim의 처리 결과 누적 (다음 claim에서 컨텍스트로 사용)
            if memory is not None:
                from structverify.core.memory import ProcessedClaimMemo
                req_summary = ""
                if schema.evidence_plan and schema.evidence_plan.requirements:
                    req_summary = ", ".join(
                        f"{r.role}@{r.time_period}"
                        for r in schema.evidence_plan.requirements
                    )
                memory.append_processed(ProcessedClaimMemo(
                    sent_id=claim.sent_id,
                    claim_text=claim.claim_text[:80],
                    indicator=schema.indicator,
                    value=schema.value,
                    unit=schema.unit,
                    time_period=schema.time_period,
                    value_role=schema.value_role.value,
                    combiner=schema.evidence_plan.combiner if schema.evidence_plan else "direct",
                    requirements_summary=req_summary,
                ))
        else:
            fail += 1
            logger.warning(f"스키마 유도 실패: {claim.sent_id}")

    logger.info(f"스키마 유도 완료: 성공 {success}건, 실패 {fail}건")

    # ─── [v6.3] 그래프 강제 반영 로직 제거 ────────────────────────────────
    # 이전 v6.2에선 그래프의 resolved time으로 schema.time_period를 덮어썼지만,
    # 한 claim에 시점이 여러 개일 때(예: "올 4월 ... 지난해 같은 달") 잘못 동작.
    # 그래프 traversal은 sentence별 "대표 시점"만 뽑는데, ratio/delta claim은
    # 두 시점이 필요하다 — 이건 evidence_plan.requirements에서 다룬다.
    #
    # 그래프는 prompt hint로만 사용 (이미 prompt 안에서 anchor_year/resolved가 주입됨).
    # LLM이 hint를 받고 time_period를 박는 결과를 신뢰.

    return claims


async def _induce_single(
    llm: LLMClient,
    claim_text: str,
    domain: str = "general",
    domain_hint: str = "",
    context: str = "",
    temporal_hint: str = "",
    memory_hint: str = "",   # [v6.4 추가]
) -> ClaimSchema | None:
    """
    단일 주장 → ClaimSchema 변환.

    Structured Outputs 사용:
      HCX: generate_structured() → HCX-007 Structured Outputs
      OpenAI: generate_structured() → response_format json_schema

    [v6.4 추가] memory_hint — 이전 처리한 claim들 컨텍스트 (지시어 해석용)
    """
    prompt = SCHEMA_INDUCTION_PROMPT.format(
        claim_text=claim_text,
        context=context or claim_text,
        domain=domain,
        domain_hint=domain_hint,
        temporal_hint=temporal_hint,
        memory_hint=memory_hint,   # [v6.4 추가]
    )

    try:
        r = await llm.generate_structured(
            prompt=prompt,
            schema=CLAIM_SCHEMA_JSON_SCHEMA,
            system_prompt="통계 분석 전문가. 정확한 정보만 추출하세요.",
        )

        # [v6.2] value_role 파싱 — LLM 응답을 enum으로 변환, 실패 시 MEASUREMENT 기본
        role_str = (r.get("value_role") or "measurement").lower().strip()
        try:
            value_role = ValueRole(role_str)
        except ValueError:
            logger.debug(f"알 수 없는 value_role={role_str!r} → MEASUREMENT 기본")
            value_role = ValueRole.MEASUREMENT

        # [v6.3] evidence_plan 파싱
        plan_raw = r.get("evidence_plan") or {}
        combiner = (plan_raw.get("combiner") or "direct").lower().strip()
        if combiner not in {"direct", "delta", "ratio_pct"}:
            combiner = "direct"

        requirements = []
        for req_raw in (plan_raw.get("requirements") or []):
            req_role = (req_raw.get("role") or "primary").strip()
            if req_role not in {"primary", "endpoint_a", "endpoint_b"}:
                continue
            requirements.append(EvidenceRequirement(
                role=req_role,
                label=req_raw.get("label"),
                indicator=req_raw.get("indicator"),
                time_period=req_raw.get("time_period"),
                population=req_raw.get("population"),
            ))

        # value_role과 combiner 일관성 보정 (LLM이 어긋나게 답했을 때 안전망)
        if value_role == ValueRole.MEASUREMENT and combiner != "direct":
            logger.debug(f"value_role=MEASUREMENT인데 combiner={combiner} → direct 강제")
            combiner = "direct"
        if value_role in (ValueRole.RATIO,) and combiner == "direct":
            combiner = "ratio_pct"
        if value_role in (ValueRole.DELTA,) and combiner == "direct":
            combiner = "delta"

        evidence_plan = EvidencePlan(
            combiner=combiner,
            requirements=requirements,
        )

        schema = ClaimSchema(
            indicator=r.get("indicator") or None,
            time_period=r.get("time_period") or None,
            unit=r.get("unit") or None,
            population=r.get("population") or None,
            value=_safe_float(r.get("value")),
            value_role=value_role,
            evidence_plan=evidence_plan,
            source_reference=r.get("source_reference") or None,
            graph_schema_candidates=r.get("graph_schema_candidates") or [],
        )

        if not _validate_schema(schema):
            logger.warning(f"스키마 유효성 미달: {claim_text[:50]}")
            return None

        return schema

    except Exception as e:
        logger.warning(f"스키마 유도 예외: {e}")
        return None


def _validate_schema(schema: ClaimSchema) -> bool:
    """indicator 없으면 KOSIS 검색 불가 → 실패 처리"""
    if not schema.indicator or len(schema.indicator.strip()) < 2:
        return False
    return True


def _safe_float(v: Any) -> float | None:
    """다양한 수치 표현 → float 변환"""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        cleaned = re.sub(r"[%,약\s]", "", v.strip())
        match = re.search(r"[\d.]+", cleaned)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
    return None