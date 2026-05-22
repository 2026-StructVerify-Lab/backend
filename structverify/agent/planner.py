"""structverify.agent.planner — Plan Agent (Phase C).

Pipeline 위치:
  Claim (with schema) → **Planner** → Plan → Loop (Phase D)

Planner 책임:
  1. claim에서 *어떤 데이터가 필요한지* 결정 (LLM 호출)
  2. *claim type* 분류 (absolute / growth_rate / diff / ratio / other)
  3. 1차 시도용 *initial steps* 제안
  4. fallback 전략 (1차 실패 시)

Reflect Agent (Phase D)는 *이 Plan을 보면서* 실제 행동 결정.
Plan은 *제안*이지 *명령*이 아님 — Reflect는 plan 무시하고 다른 step 시도 가능.

LLM client는 *callable*로 주입 (의존성 주입).
사용자 환경의 HCX/다른 LLM과 무관하게 wrap 가능.
"""
from __future__ import annotations

import json
from structverify.utils.logger import get_logger
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from pydantic import ValidationError

from .schemas import (
    ActionType,
    ClaimType,
    DataPointSpec,
    FallbackStrategy,
    Plan,
    PlanStep,
)
from .prompts import build_plan_prompt

logger = get_logger(__name__)


# ── LLM Client 인터페이스 ──────────────────────────────────────────

class LLMClient(Protocol):
    """Planner가 사용하는 LLM 호출 인터페이스.

    사용자 환경의 HCX client (또는 다른 LLM)를 *이 형태로 wrap*하면 됨.
    Phase F integration에서 실제 wiring.
    """

    async def complete(
        self,
        prompt: str,
        model: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        **kwargs: Any,
    ) -> str:
        """prompt 보내고 응답 텍스트 받기. 동기든 비동기든 OK."""
        ...


# Callable 형태도 지원 (가장 단순한 의존성 주입)
LLMCallable = Callable[[str], Awaitable[str]]


# ── 헬퍼: claim 정보 추출 ──────────────────────────────────────────

def _extract_schema_info(claim: Any) -> dict[str, Any]:
    """Claim 객체에서 schema 정보를 *dict로* 추출.

    Claim의 정확한 형태는 사용자 코드의 ClaimSchema에 의존하지만,
    pydantic dump 또는 dict-like 접근으로 *어떤 형태든* 호환.
    """
    if not claim:
        return {}

    schema = getattr(claim, "schema", None) or getattr(claim, "claim_schema", None)
    if schema is None:
        return {}

    # pydantic v2 model_dump
    if hasattr(schema, "model_dump"):
        try:
            return schema.model_dump(mode="json", exclude_none=True)
        except Exception:
            pass
    # pydantic v1 dict()
    if hasattr(schema, "dict"):
        try:
            return schema.dict(exclude_none=True)
        except Exception:
            pass
    # 평범한 dict
    if isinstance(schema, dict):
        return {k: v for k, v in schema.items() if v is not None}
    return {}


def _extract_claim_id(claim: Any) -> str:
    """Claim의 식별자 추출."""
    for attr in ("claim_id", "id", "sent_id"):
        v = getattr(claim, attr, None)
        if v:
            return str(v)
    return "unknown"


def _extract_claim_text(claim: Any) -> str:
    """Claim의 본문 텍스트 추출."""
    for attr in ("claim_text", "text", "sentence"):
        v = getattr(claim, attr, None)
        if v:
            return str(v)
    return ""


# ── JSON 추출 ─────────────────────────────────────────────────────

_JSON_PATTERNS = [
    # ```json ... ``` 또는 ``` ... ```
    re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL),
    # 그냥 { ... } (첫 번째 매칭)
    re.compile(r"(\{.*\})", re.DOTALL),
]


def _extract_balanced_json(text: str) -> str | None:
    """첫 '{' 부터 매칭되는 '}'까지 추출 (brace counting).

    정규식과 달리 응답이 잘려서 닫는 ``` 가 없거나, 중첩된 ``` 가 있어도
    동작한다. 문자열 리터럴 내부의 brace는 무시.
    """
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_json_from_response(text: str) -> dict[str, Any]:
    """LLM 응답에서 JSON 부분 추출 + 파싱.

    응답이 ```json fenced``` 또는 plain JSON, 코드 펜스 잘림 모두 처리.

    Returns:
        dict. 파싱 실패 시 빈 dict + 로그.
    """
    if not text:
        return {}

    text = text.strip()

    # 1차: 정규식 패턴들 시도
    for pattern in _JSON_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        candidate = match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            logger.debug(f"[planner] JSON 파싱 실패 ({e}), 다음 패턴 시도")
            continue

    # 2차: brace-counting fallback (코드 펜스 없이/잘림 대응)
    balanced = _extract_balanced_json(text)
    if balanced:
        try:
            result = json.loads(balanced)
            logger.info("[planner] JSON 추출: brace-counting fallback 성공")
            return result
        except json.JSONDecodeError as e:
            logger.debug(f"[planner] brace-counting JSON 파싱 실패: {e}")

    # 3차: 전체 text 시도
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"[planner] JSON 추출 실패. 응답 일부: {text[:300]!r}")
        return {}


# ── Plan 파싱 ─────────────────────────────────────────────────────

def _normalize_initial_steps(
    initial_steps: list[PlanStep],
    required_data: list[DataPointSpec],
    claim_id: str,
    fallback_query: str = "",
) -> list[PlanStep]:
    """Plan steps 정상화: catalog_search 누락 시 자동 보강.

    LLM은 종종 다음과 같이 부실한 plan을 만든다:
      A) initial_steps=[] (빈 배열)
      B) [fetch_evidence(candidate_id='<...>')] (catalog_search 빼먹음)
    두 경우 모두 loop의 _interpolate_step_input이 candidate_id placeholder를
    보간할 수 없어서 placeholder string이 그대로 KOSIS source까지 흘러간다.

    여기서 정상화:
      - case A → [catalog_search, fetch_evidence] 자동 추가
      - case B → 첫 fetch_evidence 앞에 catalog_search prepend +
                fetch의 candidate_id를 표준 placeholder로 통일

    fallback_query: required_data가 비어있을 때 catalog_search query로 쓸 문자열
                   (보통 claim_text 또는 schema.indicator).
                   비어있으면 catalog_search가 'query 비어있음' 실패 → 보간 불가.
    """
    # query: required_data의 첫 indicator 우선, 없으면 fallback_query
    query = ""
    if required_data:
        query = getattr(required_data[0], "indicator", "") or ""
    if not query:
        query = (fallback_query or "").strip()

    has_catalog = any(s.action == ActionType.CATALOG_SEARCH for s in initial_steps)
    has_fetch = any(s.action == ActionType.FETCH_EVIDENCE for s in initial_steps)

    # case A: 빈 plan
    if not initial_steps:
        logger.warning(
            f"[planner] {claim_id}: LLM이 빈 initial_steps 반환 — "
            f"[catalog_search, fetch_evidence] 자동 추가 (query={query!r})"
        )
        return [
            PlanStep(
                action=ActionType.CATALOG_SEARCH,
                input={"query": query, "top_k": 5},
                rationale="[auto-prepended] catalog_search 자동 추가",
            ),
            PlanStep(
                action=ActionType.FETCH_EVIDENCE,
                input={"candidate_id": "<catalog_search 결과의 top id>", "params": {}},
                rationale="[auto-prepended] 후보 1번 표 데이터 가져오기",
            ),
        ]

    # case B: fetch만 있고 catalog_search 없음
    if has_fetch and not has_catalog:
        logger.warning(
            f"[planner] {claim_id}: catalog_search 누락 — "
            f"fetch_evidence 앞에 자동 prepend (query={query!r})"
        )
        normalized: list[PlanStep] = []
        for s in initial_steps:
            if s.action == ActionType.FETCH_EVIDENCE:
                inp = dict(s.input or {})
                cid = inp.get("candidate_id", "")
                # placeholder-like check
                is_ph = (
                    not cid
                    or (isinstance(cid, str) and (
                        cid.startswith("<")
                        or "검색" in cid
                        or "search" in cid.lower()
                        or "결과" in cid
                    ))
                )
                if is_ph:
                    inp["candidate_id"] = "<catalog_search 결과의 top id>"
                normalized.append(PlanStep(
                    action=s.action,
                    input=inp,
                    rationale=s.rationale,
                ))
            else:
                normalized.append(s)
        return [
            PlanStep(
                action=ActionType.CATALOG_SEARCH,
                input={"query": query, "top_k": 5},
                rationale="[auto-prepended] catalog_search 자동 추가",
            ),
            *normalized,
        ]

    # case C: 정상 (catalog_search + fetch_evidence) 또는 다른 패턴 — 그대로
    return initial_steps


def _parse_plan(
    response_text: str,
    claim_id: str,
    fallback_query: str = "",
) -> Plan | None:
    """LLM 응답을 Plan 객체로 변환.

    실패 시 None 반환 + 로그. 호출자가 fallback (heuristic plan 등) 처리.

    fallback_query: required_data가 비었을 때 catalog_search query로 쓸 문자열.
    """
    data = _extract_json_from_response(response_text)
    if not data:
        return None

    # ── claim_type 파싱 ──
    raw_type = (data.get("claim_type") or "unknown").strip().lower()
    type_map = {
        "absolute": ClaimType.ABSOLUTE,
        "growth_rate": ClaimType.GROWTH_RATE,
        "difference": ClaimType.DIFFERENCE,
        "diff": ClaimType.DIFFERENCE,            # alias
        "comparison": ClaimType.COMPARISON,
        "ratio_comparison": ClaimType.COMPARISON, # alias (legacy)
        "ranking": ClaimType.RANKING,
        "aggregation": ClaimType.AGGREGATION,    # [2026-05-21] 다년 평균/총합
        "aggregate": ClaimType.AGGREGATION,      # alias
        "average": ClaimType.AGGREGATION,        # alias (LLM이 average로 출력하는 경우)
        "mean": ClaimType.AGGREGATION,           # alias
        "sum": ClaimType.AGGREGATION,            # alias
        "total": ClaimType.AGGREGATION,          # alias
        "unknown": ClaimType.UNKNOWN,
        "other": ClaimType.UNKNOWN,              # alias (legacy)
    }
    claim_type = type_map.get(raw_type, ClaimType.UNKNOWN)

    # ── required_data 파싱 ──
    raw_data = data.get("required_data") or []
    required_data: list[DataPointSpec] = []
    for item in raw_data:
        if not isinstance(item, dict):
            continue
        try:
            # role은 schema에 없을 수도 있으니 안전하게
            spec_kwargs = {
                "indicator": str(item.get("indicator", "")).strip(),
                "time": str(item.get("time", "")).strip(),
                "population": item.get("population") or None,
                "unit_hint": item.get("unit_hint") or item.get("unit") or None,
            }
            # 비어있는 필수 필드면 스킵
            if not spec_kwargs["indicator"] or not spec_kwargs["time"]:
                logger.debug(f"[planner] data point 스킵 (필수 필드 누락): {item}")
                continue
            spec = DataPointSpec(**spec_kwargs)
            required_data.append(spec)
        except (ValidationError, TypeError) as e:
            logger.debug(f"[planner] DataPointSpec 파싱 실패: {item} | {e}")

    # ── initial_steps 파싱 ──
    raw_steps = data.get("initial_steps") or []
    initial_steps: list[PlanStep] = []
    for item in raw_steps:
        if not isinstance(item, dict):
            continue
        action_str = (item.get("action") or "").strip().lower()
        try:
            action = ActionType(action_str)
        except ValueError:
            logger.debug(f"[planner] 알 수 없는 action: {action_str!r}, 스킵")
            continue
        try:
            step = PlanStep(
                action=action,
                input=item.get("input") or {},
                rationale=str(item.get("rationale") or "").strip(),
            )
            initial_steps.append(step)
        except (ValidationError, TypeError) as e:
            logger.debug(f"[planner] PlanStep 파싱 실패: {item} | {e}")

    # ── fallback 파싱 ──
    raw_fallback = data.get("fallback") or {}
    if not isinstance(raw_fallback, dict):
        raw_fallback = {}
    try:
        fallback = FallbackStrategy(
            use_original_text=bool(raw_fallback.get("use_original_text", False)),
            alternative_keywords=list(raw_fallback.get("alternative_keywords") or []),
            give_up_after_attempts=int(raw_fallback.get("give_up_after_attempts", 5)),
        )
    except (ValidationError, TypeError, ValueError) as e:
        logger.debug(f"[planner] FallbackStrategy 파싱 실패: {e}")
        fallback = FallbackStrategy()

    # ── ★ initial_steps 정상화 (LLM의 부실한 plan 보강) ──
    initial_steps = _normalize_initial_steps(
        initial_steps, required_data, claim_id, fallback_query=fallback_query
    )

    # ── Plan 생성 ──
    try:
        plan = Plan(
            claim_id=claim_id,
            claim_type=claim_type,
            required_data=required_data,
            initial_steps=initial_steps,
            fallback=fallback,
            calculation_formula=(data.get("calculation_formula") or None) or None,
            notes=str(data.get("notes") or "").strip() or None,
        )
        return plan
    except ValidationError as e:
        logger.warning(f"[planner] Plan 최종 validation 실패: {e}")
        return None


# ── Heuristic Fallback Plan ──────────────────────────────────────

def _heuristic_plan(claim: Any, claim_id: str) -> Plan:
    """LLM 호출 실패 시 *최소한의 Plan*을 만들어 loop이 멈추지 않게.

    - claim_type=other
    - schema가 있으면 데이터 점 1개 (current role)
    - 첫 step: catalog_search (indicator 기반)
    """
    schema_info = _extract_schema_info(claim)
    claim_text = _extract_claim_text(claim)

    required_data: list[DataPointSpec] = []
    if schema_info.get("indicator"):
        try:
            required_data.append(DataPointSpec(
                indicator=str(schema_info["indicator"]),
                time=str(schema_info.get("time_period") or ""),
                population=schema_info.get("population") or None,
                unit_hint=schema_info.get("unit") or None,
            ))
        except (ValidationError, TypeError):
            pass

    # 추측 검색어
    query_terms = []
    if schema_info.get("indicator"):
        query_terms.append(str(schema_info["indicator"]))
    if schema_info.get("parent_path"):
        # parent_path가 "인구 > 출생 > 출생아 수" 같은 형식
        parts = [p.strip() for p in str(schema_info["parent_path"]).split(">") if p.strip()]
        if len(parts) >= 1:
            query_terms.append(parts[0])

    query = " ".join(query_terms) if query_terms else (claim_text[:30] if claim_text else "통계")

    initial_steps = [
        PlanStep(
            action=ActionType.CATALOG_SEARCH,
            input={"query": query, "top_k": 5},
            rationale="(heuristic fallback) indicator 기반 1차 검색",
        ),
        PlanStep(
            action=ActionType.FINISH,
            input={
                "verdict": "unverifiable",
                "confidence": 0.3,
                "explanation": "Plan Agent가 LLM 호출에 실패하여 휴리스틱 fallback. 자세한 검증 불가.",
            },
            rationale="(heuristic fallback) plan 부재로 최종 unverifiable",
        ),
    ]

    return Plan(
        claim_id=claim_id,
        claim_type=ClaimType.UNKNOWN,
        required_data=required_data,
        initial_steps=initial_steps,
        fallback=FallbackStrategy(),
        notes="heuristic fallback plan (LLM 미사용 또는 실패)",
    )


# ── Planner 본체 ──────────────────────────────────────────────────

@dataclass
class PlannerConfig:
    """Planner 동작 설정."""

    model: str = "HCX-007"
    """LLM 모델 이름. config.agent.llm.plan_model 에서 가져오면 됨."""

    temperature: float = 0.1
    """낮을수록 결정적. plan은 결정적이 좋음 → 0.1 권장."""

    max_tokens: int = 4000

    max_retries: int = 1
    """JSON 파싱 실패 시 LLM 재호출 횟수. 0이면 fallback 즉시."""


class Planner:
    """Plan Agent.

    Usage:
        planner = Planner(llm_call=my_llm_call, config=PlannerConfig(model="HCX-007"))
        plan = await planner.plan(claim, source_text=article_text, anchor_year=2025)
        workspace.write_plan(claim.claim_id, plan.model_dump(mode="json"))
    """

    def __init__(
        self,
        llm_call: LLMCallable | None = None,
        config: PlannerConfig | None = None,
    ):
        self.llm_call = llm_call
        self.config = config or PlannerConfig()

    async def plan(
        self,
        claim: Any,
        source_text: str | None = None,
        anchor_year: int | str | None = None,
    ) -> Plan:
        """Claim → Plan.

        Args:
            claim: structverify의 Claim 객체. .claim_text + .schema + .claim_id 속성 가정.
            source_text: 원문 기사 전체 (옵션 — prompt에 일부 삽입).
            anchor_year: 문서 anchor_year (옵션 — 시점 해소용).

        Returns:
            Plan. LLM 호출 실패 시 *heuristic fallback Plan*.
        """
        claim_id = _extract_claim_id(claim)
        claim_text = _extract_claim_text(claim)
        schema_info = _extract_schema_info(claim)

        if not claim_text:
            logger.warning(f"[planner] {claim_id}: claim_text 비어있음, heuristic fallback")
            return _heuristic_plan(claim, claim_id)

        if self.llm_call is None:
            logger.warning(f"[planner] {claim_id}: llm_call 미주입, heuristic fallback")
            return _heuristic_plan(claim, claim_id)

        prompt = build_plan_prompt(
            claim_text=claim_text,
            schema_info=schema_info or None,
            source_excerpt=source_text,
            anchor_year=anchor_year,
        )
        logger.info(
            f"[planner] {claim_id}: prompt 구성 완료 ({len(prompt)}자). "
            f"schema={'있음' if schema_info else '없음'} source={'있음' if source_text else '없음'}"
        )

        # LLM 호출 (재시도 포함)
        last_response = ""
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self.llm_call(prompt)
                last_response = response or ""
                logger.info(
                    f"[planner] {claim_id}: LLM 응답 받음 ({len(last_response)}자) "
                    f"[시도 {attempt + 1}/{self.config.max_retries + 1}]"
                )
            except Exception as e:
                logger.warning(
                    f"[planner] {claim_id}: LLM 호출 실패 [시도 {attempt + 1}]: "
                    f"{type(e).__name__}: {e}"
                )
                continue

            # fallback query: schema.indicator > claim_text 앞부분
            fallback_query = ""
            if isinstance(schema_info, dict):
                fallback_query = (schema_info.get("indicator") or "").strip()
            if not fallback_query and claim_text:
                # claim_text 앞 40자 정도까지 (너무 길면 search 품질 떨어짐)
                fallback_query = claim_text.strip()[:40]

            plan = _parse_plan(last_response, claim_id, fallback_query=fallback_query)
            if plan is not None:
                # [2026-05-21] value_role 후처리 — schema_inductor가 분기한 *역할*과
                # LLM이 만든 claim_type이 불일치하면 *value_role을 신뢰*하고 정정.
                # LLM이 같은 claim_text의 sub-claim들을 동일 plan_type으로 잘못
                # 분류하던 버그(2026-05-21 진단: 출생아 수 base + 증가율 둘 다
                # growth_rate)를 결정론적으로 차단.
                _role = (schema_info or {}).get("value_role") if isinstance(schema_info, dict) else None
                _role_to_type = {
                    "base": ClaimType.ABSOLUTE,
                    "derived_rate": ClaimType.GROWTH_RATE,
                    "derived_difference": ClaimType.DIFFERENCE,
                    # [2026-05-21] 다년 집계 — 도메인 무관, schema_inductor가 분기
                    "aggregation": ClaimType.AGGREGATION,
                }
                _expected_type = _role_to_type.get(_role)
                if _expected_type and plan.claim_type != _expected_type:
                    logger.info(
                        f"[planner] {claim_id}: value_role={_role!r} 기반 정정 — "
                        f"LLM type={plan.claim_type.value} → {_expected_type.value}"
                    )
                    plan = plan.model_copy(update={"claim_type": _expected_type})
                logger.info(
                    f"[planner] {claim_id}: Plan 생성 완료. "
                    f"type={plan.claim_type.value}, data_points={len(plan.required_data)}, "
                    f"steps={len(plan.initial_steps)}, formula={plan.calculation_formula!r}"
                )
                return plan

            logger.warning(
                f"[planner] {claim_id}: Plan 파싱 실패 [시도 {attempt + 1}]. "
                f"응답 일부: {last_response[:200]!r}"
            )

        # 모든 시도 실패 → fallback
        logger.warning(f"[planner] {claim_id}: 모든 시도 실패. heuristic fallback 사용.")
        return _heuristic_plan(claim, claim_id)


# ── 편의 함수 ──────────────────────────────────────────────────────

async def build_plan(
    claim: Any,
    llm_call: LLMCallable | None = None,
    source_text: str | None = None,
    anchor_year: int | str | None = None,
    config: PlannerConfig | None = None,
) -> Plan:
    """일회성 Plan 생성 (Planner 인스턴스 안 만들고).

    Phase D Loop에서 한 번씩만 호출하면 되니 충분.
    """
    planner = Planner(llm_call=llm_call, config=config)
    return await planner.plan(claim, source_text=source_text, anchor_year=anchor_year)