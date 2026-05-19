"""
structverify.agent.tools.fetch_evidence — 데이터 조회 Tool.

catalog_search로 *후보 발견* → fetch_evidence로 *실제 수치 조회*.

작동:
  1. context.datasources에서 source 선택 (catalog_search와 동일 source 권장)
  2. context.claim.schema에서 indicator/time_period/population/unit 추출해 params 보강
  3. source.fetch_evidence(candidate_id, params) 호출
  4. EvidenceData 반환 + workspace observation 저장

source-specific 파라미터:
  - KOSIS: {"prdSe": "M", "startPrdDe": "202504", "endPrdDe": "202504", ...}
  - Custom CSV: {"row_filter": "month=4 AND year=2025", "column": "births"}
  - 외부 API: provider별 다름

Agent는 *params를 모르면 빈 dict {}로 호출*. claim.schema에서 자동 보강됨.
"""
from __future__ import annotations

from structverify.utils.logger import get_logger
from typing import Any

from ..schemas import ActionType
from .base import ToolBase, ToolContext, ToolResult, register_tool

logger = get_logger(__name__)


@register_tool(ActionType.FETCH_EVIDENCE)
class FetchEvidenceTool(ToolBase):
    """카탈로그 후보의 실제 수치 데이터 조회.

    catalog_search로 candidate_id 알아낸 후 호출.
    """

    name = ActionType.FETCH_EVIDENCE
    description = (
        "데이터 소스에서 *실제 수치* 조회. catalog_search로 candidate_id 알아낸 후 호출. "
        "params는 source별 다름 (KOSIS는 시점 필터 등). 모르면 빈 dict {} 전달 — "
        "claim.schema에서 자동 보강됨."
    )
    input_schema = {
        "candidate_id": "catalog_search 결과의 id. 예: 'DT_1B8000G'",
        "params": "(선택) source별 파라미터 dict. 모르면 {}",
        "source": "(선택) 데이터 소스 이름. catalog_search와 동일하게.",
    }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        candidate_id = (input_data.get("candidate_id") or "").strip()
        if not candidate_id:
            return ToolResult(
                output={},
                summary="실패: candidate_id 비어있음",
                success=False,
                error="candidate_id는 비어있을 수 없습니다.",
            )

        params = input_data.get("params") or {}
        if not isinstance(params, dict):
            return ToolResult(
                output={},
                summary=f"실패: params는 dict이어야 함, got {type(params).__name__}",
                success=False,
                error="params는 dict 또는 None이어야 합니다.",
            )

        # ★ ADD: claim.schema에서 누락된 params 자동 보강
        # Planner LLM은 KOSIS spec을 모르므로 params를 비워둠 → 여기서 보강
        # claim.schema에는 schema_inductor가 추출한 indicator/time_period/unit/population 들어있음
        claim = getattr(context, "claim", None)
        schema = getattr(claim, "schema", None) if claim is not None else None
        if schema is not None:
            # dict 형태로 변환해서 안전하게 접근
            params = dict(params)
            if not params.get("indicator") and getattr(schema, "indicator", None):
                params["indicator"] = schema.indicator
            if not params.get("time_period") and getattr(schema, "time_period", None):
                params["time_period"] = schema.time_period
            if not params.get("population") and getattr(schema, "population", None):
                params["population"] = schema.population
            if not params.get("unit_hint") and getattr(schema, "unit", None):
                params["unit_hint"] = schema.unit
            # ── [v6.17] growth_rate 직접계산용 — fetch 범위 확장 ──────────
            # claim에 prev_time_period가 있으면(증가율/변화량 claim),
            # startPrdDe를 prev 시점까지 당겨서 현재+이전 시점을 한 번에
            # 받아온다. 그래야 loop이 같은 표 rows에서 prev 값을 찾아
            # 증가율을 직접 계산할 수 있음. (1회 fetch로 두 해 확보)
            prev_tp = getattr(schema, "prev_time_period", None)
            cur_tp = params.get("time_period")
            if prev_tp and cur_tp and not params.get("startPrdDe"):
                # 'YYYY-MM'/'YYYY' → 비교해서 더 이른 쪽을 start로
                _p = str(prev_tp).replace("-", "").strip()
                _c = str(cur_tp).replace("-", "").strip()
                if _p and _c and _p.isdigit() and _c.isdigit():
                    start_raw, end_raw = (prev_tp, cur_tp) if _p <= _c else (cur_tp, prev_tp)
                    params["_range_start"] = str(start_raw)
                    params["_range_end"] = str(end_raw)
                    logger.info(
                        f"[fetch_evidence] growth_rate fetch 범위 확장: "
                        f"{start_raw} ~ {end_raw} (prev={prev_tp}, current={cur_tp})"
                    )
            logger.info(
                f"[fetch_evidence] claim.schema에서 params 보강: "
                f"indicator={params.get('indicator')!r} "
                f"time_period={params.get('time_period')!r} "
                f"population={params.get('population')!r} "
                f"unit_hint={params.get('unit_hint')!r}"
            )

        # ── [v6.22] schema/params 없으면 fetch 거부 ──────────────────
        # indicator가 없으면 connector가 '무엇을' 조회할지 자체를 모르고
        # 기본값(drows[0])을 반환 → 통합·연간 행이 그대로 새어나온다.
        # period guard·통합행 거부 모두 indicator를 근거로 동작하므로
        # indicator가 비면 작동하지 못한다.
        # 예: 기사 제목 claim은 schema 유도 실패 → schema=없음 →
        #     planner가 시점만 추측 → '출생사망혼인이혼 238317' 누수.
        # time_period만 있고 indicator가 없으면 fetch하지 않고 거부한다.
        if not params.get("indicator"):
            logger.warning(
                f"[fetch_evidence] indicator 없음 → fetch 거부: "
                f"candidate={candidate_id} (params={ {k: v for k, v in params.items() if k in ('indicator', 'time_period', 'population')} }) "
                f"— connector 기본값 누수 방지. claim에 검증 가능 수치가 "
                f"없거나 schema 유도가 실패한 claim."
            )
            return ToolResult(
                output={"evidence": None, "reason": "no_indicator"},
                summary=(
                    "fetch 거부: claim에 indicator 없음 "
                    "(schema 유도 실패 claim — 검증 불가)"
                ),
                success=False,
                error="claim.schema에 indicator가 없어 fetch 대상을 특정할 수 없음",
            )

        # source 선택
        ds_config = context.config.get("data_sources", {}) if context.config else {}
        default_source = ds_config.get("default_source", "kosis")
        source_name = (input_data.get("source") or default_source).strip()

        # ── [v6.21] verified_facts 캐시 조회 ──────────────────────────
        # 같은 (indicator, time_period)를 다른 claim이 이미 검증했으면
        # catalog_search + fetch 전체를 건너뛰고 저장된 수치를 재사용한다.
        # 예: "올해 출생아 수 20,717명" 검증 후 → "작년 대비 8.7% 증가"
        #     claim이 올해값을 재검색 없이 즉시 가져옴.
        _cache_ind = params.get("indicator")
        _cache_tp = params.get("time_period")
        _cache_unit = params.get("unit_hint")
        _ws = getattr(context, "workspace", None)
        if _ws is not None and _cache_ind and _cache_tp:
            try:
                hit = _ws.lookup_verified_fact(
                    _cache_ind, _cache_tp, unit_hint=_cache_unit
                )
            except Exception:
                hit = None
            if hit is not None:
                logger.info(
                    f"[fetch_evidence] verified_facts 캐시 적중 — "
                    f"indicator={_cache_ind!r} time={_cache_tp!r} "
                    f"value={hit.get('value')} (재검색 생략)"
                )
                cached_evidence = {
                    "value": hit.get("value"),
                    "unit": hit.get("unit", "") or "",
                    "time_period": hit.get("time_period", "") or "",
                    "source": hit.get("source", "KOSIS") or "KOSIS",
                    "stat_table_id": "",
                    "stat_name": "(verified_facts 캐시)",
                    "rows": [],
                    "raw": {"from_cache": True, "origin_claim": hit.get("claim_id")},
                    "from_verified_cache": True,
                }
                return ToolResult(
                    output={"evidence": cached_evidence, "used_candidate_id": "cache"},
                    summary=(
                        f"verified_facts 캐시 재사용: {_cache_ind} "
                        f"{_cache_tp} = {hit.get('value')}{hit.get('unit', '')}"
                    ),
                    success=True,
                )

        source = context.datasources.get(source_name) if context.datasources else None
        if source is None:
            available = list(context.datasources.keys()) if context.datasources else []
            return ToolResult(
                output={"requested_source": source_name, "available": available},
                summary=f"실패: source={source_name!r} 등록 안 됨",
                success=False,
                error=(
                    f"DataSource '{source_name}'이 context.datasources에 없습니다. "
                    f"가능한 source: {available}"
                ),
            )

        # ── [v6.18] 후보 순회 fetch ──────────────────────────────────
        # top 후보가 무관한 표(관련성 체크 거부)거나 데이터 없음이면
        # _candidate_fallbacks의 다음 후보로 재시도. 최대 4개까지 시도.
        fallback_ids = input_data.get("_candidate_fallbacks") or []
        try_ids = [candidate_id] + [
            fid for fid in fallback_ids if fid and fid != candidate_id
        ]
        try_ids = try_ids[:4]  # top + fallback 3개

        evidence = None
        used_id = candidate_id
        last_err: str | None = None
        for idx, try_id in enumerate(try_ids):
            try:
                ev = await source.fetch_evidence(
                    candidate_id=try_id, params=params,
                )
            except Exception as e:
                logger.warning(
                    f"[fetch_evidence] 후보 {try_id} fetch 예외: "
                    f"{type(e).__name__}: {e}"
                )
                last_err = f"{type(e).__name__}: {e}"
                continue
            if ev is not None:
                evidence = ev
                used_id = try_id
                if idx > 0:
                    logger.info(
                        f"[fetch_evidence] top 후보 실패 → 후보 {idx+1}번째 "
                        f"{try_id} 로 성공"
                    )
                break
            else:
                logger.info(
                    f"[fetch_evidence] 후보 {try_id} → None "
                    f"(관련성 거부/데이터 없음), 다음 후보 시도"
                )
        candidate_id = used_id

        # evidence None 처리 (모든 후보 실패)
        if evidence is None:
            return ToolResult(
                output={"source": source_name, "candidate_id": candidate_id,
                        "params": params, "evidence": None,
                        "tried_candidates": try_ids},
                summary=(
                    f"fetch({source_name}): 후보 {len(try_ids)}개 모두 실패 "
                    f"(관련 표 없음)"
                ),
                success=False,
                error=(
                    last_err
                    or "모든 catalog 후보가 관련성 체크 실패 또는 데이터 없음."
                ),
            )

        # 결과 정규화 (EvidenceData는 dict 호환)
        evidence_dict = dict(evidence) if hasattr(evidence, "items") else {}

        # workspace observation 저장
        try:
            obs_name = f"iter{context.iter_num:03d}_fetch_{candidate_id}"
            context.workspace.write_observation(
                context.claim_id,
                obs_name,
                {
                    "source": source_name,
                    "candidate_id": candidate_id,
                    "params": params,
                    "evidence": evidence_dict,
                },
            )
        except Exception as e:
            logger.debug(f"[fetch_evidence] observation 저장 실패: {e}")

        # 요약
        value = evidence_dict.get("value")
        unit = evidence_dict.get("unit", "")
        time_period = evidence_dict.get("time_period", "")
        rows = evidence_dict.get("rows", [])
        rows_info = f", rows={len(rows)}" if isinstance(rows, list) and rows else ""
        summary = (
            f"fetch({source_name}, {candidate_id}): value={value!r} unit={unit!r} "
            f"time={time_period!r}{rows_info}"
        )

        return ToolResult(
            output={
                "source": source_name,
                "candidate_id": candidate_id,
                "params": params,
                "evidence": evidence_dict,
            },
            summary=summary,
            success=True,
        )