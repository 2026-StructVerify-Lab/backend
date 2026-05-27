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


def _autoload_fallback_candidate_ids(
    workspace, claim_id, current_id: str, limit: int = 4
) -> list[str]:
    """LLM이 input에 `_candidate_fallbacks`를 안 넘긴 경우, workspace의
    가장 최근 catalog_search observation들에서 다른 후보 ids를 자동 추출.

    이 자동 주입이 없으면 fetch_evidence가 top 후보 1개만 시도하고
    실패하면 reflect가 catalog_search를 반복 호출 → 중복차단 → 강제
    unverifiable로 죽음. 후보 5개 중 정확한 표가 2~5순위에 있으면
    영영 도달 못 함.

    [패치 C] 현재 claim의 catalog observation에서 충분한 후보를 못
    얻으면 같은 job의 다른 claim들의 catalog observation에서도 후보를
    수집해 합친다. 같은 KOSIS 표가 여러 지표를 가진 케이스에서, 한
    claim의 검색이 우연히 정답 표를 top으로 못 잡았더라도 다른 claim의
    catalog가 그 표를 후보로 가졌으면 활용 가능.
    """
    seen = {current_id}
    out: list[str] = []

    def _collect_from(claim_cid: str) -> None:
        try:
            names = workspace.list_observations(claim_cid)
        except Exception:
            return
        cat_names = sorted(
            [n for n in names if "catalog_search" in n.lower()],
            reverse=True,
        )
        for name in cat_names:
            data = workspace.read_observation(claim_cid, name)
            if not isinstance(data, dict):
                continue
            cands = (data.get("output") or {}).get("candidates") or []
            for c in cands:
                cid = c.get("id") if isinstance(c, dict) else None
                if not cid or cid in seen:
                    continue
                out.append(cid)
                seen.add(cid)
                if len(out) >= limit:
                    return

    # 1차: 현재 claim의 catalog observations
    _collect_from(claim_id)
    if len(out) >= limit:
        return out
    # 2차: 같은 job의 다른 claim들 — 표 다양성 확보
    try:
        other_cids = [c for c in workspace.list_claims() if c != str(claim_id)]
    except Exception:
        other_cids = []
    for other in other_cids:
        if len(out) >= limit:
            break
        _collect_from(other)
    return out


def _collect_candidate_pool(
    workspace, claim_id, current_id: str, limit: int = 20,
) -> list[dict]:
    """catalog_search + explore_catalog observation에서 후보 표 dict 수집.

    각 후보 dict는 {id, name, score, category_path, raw?, _pool_source} 형태.
    탈중복 by id. catalog_ranker가 메타데이터까지 보고 의미 점수를 매기기 위함.

    수집 순서:
      1. 현재 claim의 catalog_search candidates (cosine recall — 점수 보존)
      2. 현재 claim의 explore_catalog example tables (categorical recall — 다른 path)
      3. 같은 job 다른 claim의 catalog candidates (job-level diversity)
    """
    seen: set[str] = set()
    if current_id:
        seen.add(current_id)
    pool: list[dict] = []

    def _add(cand: dict, source: str) -> bool:
        cid = str(cand.get("id") or "").strip()
        if not cid or cid in seen:
            return len(pool) < limit
        seen.add(cid)
        out = dict(cand)
        out["_pool_source"] = source
        pool.append(out)
        return len(pool) < limit

    def _collect_catalog(claim_cid: str) -> None:
        try:
            names = workspace.list_observations(claim_cid)
        except Exception:
            return
        cat_names = sorted(
            [n for n in names if "catalog_search" in n.lower()],
            reverse=True,
        )
        for name in cat_names:
            data = workspace.read_observation(claim_cid, name)
            if not isinstance(data, dict):
                continue
            cands = (data.get("output") or {}).get("candidates") or []
            for c in cands:
                if isinstance(c, dict):
                    if not _add(c, "catalog"):
                        return

    def _collect_explore(claim_cid: str) -> None:
        try:
            names = workspace.list_observations(claim_cid)
        except Exception:
            return
        ex_names = sorted(
            [n for n in names if "explore_catalog" in n.lower()],
            reverse=True,
        )
        for name in ex_names:
            data = workspace.read_observation(claim_cid, name)
            if not isinstance(data, dict):
                continue
            cats = data.get("categories") or []
            for cat in cats:
                if not isinstance(cat, dict):
                    continue
                category_label = cat.get("category_label", "")
                for ex in (cat.get("examples") or []):
                    if isinstance(ex, dict) and ex.get("stat_id"):
                        if not _add({
                            "id": ex.get("stat_id"),
                            "name": ex.get("stat_name", ""),
                            "score": 0.0,
                            "category_path": category_label,
                        }, "explore"):
                            return

    _collect_catalog(claim_id)
    _collect_explore(claim_id)

    if len(pool) < limit:
        try:
            other_cids = [c for c in workspace.list_claims() if c != str(claim_id)]
        except Exception:
            other_cids = []
        for other in other_cids:
            if len(pool) >= limit:
                break
            _collect_catalog(other)

    return pool


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
        "candidate_id": "catalog_search 결과의 id",
        "params": (
            "(선택) source별 파라미터 dict. indicator/time_period/population은 "
            "claim.schema에서 자동 보강됨. 핵심 옵션 match_criteria: 직전 fetch의 "
            "row sample 컬럼명을 본 뒤 어떤 컬럼이 어떤 값과 매칭돼야 하는지 dict로 "
            "명시. row 매칭이 모든 criteria 만족 row로 좁혀짐. "
            "형식 예: {\"match_criteria\": {\"<column_name>\": \"<expected_substring>\"}}. "
            "컬럼명은 row sample에 노출된 키를 그대로 사용 — 도메인 무관."
        ),
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
            # ── [L 패치 2026-05-21] population은 schema 강제 덮어쓰기 ──
            # LLM이 claim_text 전체(여러 region 등장)를 보고 sub-claim의 schema와
            # *다른 region*을 fetch에 박는 케이스가 잦음 (의료장비 서울 claim에
            # population='강원도' 박아 강원 row 1336 가져옴 → evidence pool 오염
            # → LLM이 finish할 때 헷갈려 잘못된 mismatch).
            # sub-claim의 정체성(population)은 schema가 진실. LLM이 다른 값을
            # 명시했더라도 schema 우선으로 덮어씀. schema 값이 "전체"/None 같은
            # 비특정 값이면 LLM 값 유지.
            # 주의: time_period는 *growth_rate/difference에서 LLM이 prev 시점을
            # 의도적으로* 박아야 하므로 덮어쓰지 *않음* (기존 누락 보강만).
            _sch_pop = (getattr(schema, "population", None) or "").strip()
            if _sch_pop and _sch_pop not in ("전체", "전국", "계", "total"):
                _llm_pop = (params.get("population") or "").strip()
                if _llm_pop and _llm_pop != _sch_pop:
                    logger.info(
                        f"[fetch_evidence] population LLM={_llm_pop!r} → "
                        f"schema {_sch_pop!r} 덮어씀 (sub-claim 정체성 우선)"
                    )
                params["population"] = _sch_pop
            if not params.get("unit_hint") and getattr(schema, "unit", None):
                params["unit_hint"] = schema.unit
            # [P32 2026-05-22] LLM 기반 relevance fallback이 활용할 컨텍스트.
            # raw_claim(원문 문장)과 parent_path(계층 카테고리)를 params에 실어
            # KOSISDataSource의 v6.17 가드에서 LLM judge에 전달.
            if not params.get("parent_path") and getattr(schema, "parent_path", None):
                params["parent_path"] = schema.parent_path
            if not params.get("raw_claim"):
                _claim_text = getattr(claim, "claim_text", None)
                if _claim_text:
                    params["raw_claim"] = str(_claim_text)[:400]
            # [패치] derived claim (~증가율 등)의 unit_hint='%'는 KOSIS 표의
            # base 단위 row(명/건) 매칭을 막아 evidence 0건 → unverifiable로
            # 죽이는 원인. derived claim에서는 fetch 시 base row를 받아야
            # loop의 growth_rate/difference 직접계산 경로가 작동한다.
            # claim.schema.indicator(원본)에 derived suffix가 있으면
            # unit_hint를 비워 _select_best_row의 unit 가드를 우회한다.
            #
            # [P27 2026-05-22] suffix list 확장 + *indicator unwrap*.
            # "X 증가 수", "X 감소 수", "X 증가량" 등 *공백 포함 두 단어 표현*도 derived
            # 류로 인식. derived 감지 시 fetch params의 indicator를 *원지표(X)*로
            # unwrap해서 KOSIS 검색. KOSIS는 "X 증가 수" 자체 row가 거의 없으니
            # 원지표 row를 받아 loop이 (cur - prev) 직접 계산하도록 유도.
            #
            # 한국어 동사형 변화 표현 — 도메인 무관 일반 패턴 (의료/인구/경제 공통).
            _DERIVED_RATE_SUFFIXES = (
                "증가율", "감소율", "증감률", "변화율", "상승률", "하락률",
                "비율", "비중",
            )
            _DERIVED_DIFF_SUFFIXES = (
                "증가 수", "감소 수", "증감 수",
                "증가량", "감소량", "증감량",
                "증가폭", "감소폭",
                "신규 도입 수", "도입 수",
                "신규 수", "추가 수",
                "증가", "감소", "증감", "변화", "차이",  # 짧은 형태 (K 패치 기존)
            )
            _claim_ind = (getattr(schema, "indicator", "") or "").strip()
            _matched_suffix: str | None = None
            _is_rate = False
            for _sfx in _DERIVED_RATE_SUFFIXES:
                if _claim_ind.endswith(_sfx):
                    _matched_suffix = _sfx
                    _is_rate = True
                    break
            if not _matched_suffix:
                # 긴 것 먼저 (예: "증가 수" 가 "증가"보다 먼저 매칭되도록 정렬은 list 순서로 보장)
                for _sfx in _DERIVED_DIFF_SUFFIXES:
                    if _claim_ind.endswith(_sfx):
                        _matched_suffix = _sfx
                        break

            if _matched_suffix:
                # (1) unit_hint='%' 제거 (rate 케이스)
                if _is_rate and params.get("unit_hint"):
                    logger.info(
                        f"[fetch_evidence] derived rate claim '{_claim_ind}' — "
                        f"unit_hint={params.get('unit_hint')!r} 제거 "
                        f"(base 단위 row 매칭 위해)"
                    )
                    params.pop("unit_hint", None)
                # (2) indicator unwrap — KOSIS 검색용 원지표 추출
                _root_indicator = _claim_ind[: -len(_matched_suffix)].rstrip()
                if _root_indicator and _root_indicator != _claim_ind:
                    logger.info(
                        f"[fetch_evidence] derived indicator unwrap: "
                        f"'{_claim_ind}' → '{_root_indicator}' (suffix={_matched_suffix!r}) — "
                        f"KOSIS는 보통 *원지표 row*만 제공, 차이/증가율은 loop이 직접 계산"
                    )
                    params["indicator"] = _root_indicator
                    # rate 케이스도 unit_hint 비움 (위에서 처리했지만 안전)
                    if not _is_rate:
                        # derived_difference 케이스: 단위는 원지표 그대로 (예: 대/명/건)
                        # → unit_hint를 schema.unit으로 유지 (이미 위에서 채움)
                        pass
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

            # [2026-05-21] match_criteria carry-over 가드 — reflect LLM이 직전 claim의
            # matched_row에서 criteria를 복사해 넘기는 경우가 있어, 다른 sub-claim
            # (population='인천')인데 criteria={'시군구': '강원도'} 같은 충돌이 발생.
            # 22:41:28 인천 claim 예: schema.population='인천'인데 LLM criteria='강원도'
            # → _select_best_row가 '강원' substring 매칭 시도 → 모든 후보 매칭 실패.
            # 가드: schema.population이 *구체적*이고 (전체/전국 등 제외), match_criteria의
            # 어떤 value에도 schema.population과 양방향 substring 매칭이 *전혀* 없으면
            # → criteria 폐기 (LLM의 carry-over로 간주).
            _sch_pop_norm = _sch_pop  # 위에서 정의된 schema.population
            _criteria = params.get("match_criteria")
            if (
                _sch_pop_norm
                and _sch_pop_norm not in ("전체", "전국", "계", "total")
                and isinstance(_criteria, dict) and _criteria
            ):
                _has_overlap = False
                for _cv in _criteria.values():
                    _cv_s = str(_cv or "").strip()
                    if not _cv_s:
                        continue
                    if _cv_s in _sch_pop_norm or _sch_pop_norm in _cv_s:
                        _has_overlap = True
                        break
                if not _has_overlap:
                    logger.warning(
                        f"[fetch_evidence] match_criteria carry-over 가드: "
                        f"schema.population={_sch_pop_norm!r}와 충돌하는 "
                        f"criteria={_criteria!r} 폐기 (LLM이 직전 claim 정보 복사 의심)"
                    )
                    params.pop("match_criteria", None)

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
        _cache_pop = params.get("population")
        _ws = getattr(context, "workspace", None)
        if _ws is not None and _cache_ind and _cache_tp:
            try:
                # [2026-05-21] population까지 키에 포함 — 같은 (indicator, time)이라도
                # 다른 지역 sub-claim의 캐시 값이 적중하던 버그(22:54 트레이스 — 서울
                # sub-claim이 강원도/197 같은 다른 값 받음) 차단.
                hit = _ws.lookup_verified_fact(
                    _cache_ind, _cache_tp,
                    unit_hint=_cache_unit, population=_cache_pop,
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
        # _candidate_fallbacks의 다음 후보로 재시도. 최대 5개까지 시도.
        fallback_ids = input_data.get("_candidate_fallbacks") or []
        # ── i'' 패치: LLM이 _candidate_fallbacks를 안 넘긴 케이스 처리 ──
        # 비어있으면 workspace의 직전 catalog_search observation에서
        # 후보 ids를 자동 추출. 안 그러면 top 후보 1개만 시도하고 죽음.
        # (project_fetch_lockup — reflect의 catalog_search 무한 반복)
        if not fallback_ids and context.workspace is not None:
            auto = _autoload_fallback_candidate_ids(
                context.workspace, context.claim_id, candidate_id, limit=4
            )
            if auto:
                fallback_ids = auto
                logger.info(
                    f"[fetch_evidence] _candidate_fallbacks 자동 주입: "
                    f"{auto} (LLM이 안 넘김 → catalog observation에서 추출)"
                )
        # ── [패치 A] job 안에서 이미 fetch 성공한 stat_id를 1순위 fallback ──
        # 같은 KOSIS 표가 여러 지표(출생아 수/합계출산율/혼인 건수)를 같이
        # 갖고 있는데 catalog는 검색어별로 다른 표를 top으로 주는 경우 대응.
        # 다른 claim이 표 X에서 성공했다면, 현재 claim의 top 후보가 부적절
        # 해도 표 X를 우선 시도한다. (candidate_id 자체가 표 X면 영향 없음.)
        prior_success_ids: list[str] = []
        if context.workspace is not None:
            try:
                prior_success_ids = context.workspace.read_successful_stat_ids()
            except Exception as e:
                logger.debug(f"[fetch_evidence] successful_stat_ids 읽기 실패: {e}")
                prior_success_ids = []
        if prior_success_ids:
            logger.info(
                f"[fetch_evidence] 직전 success stat_id 우선 시도: "
                f"{prior_success_ids} (job 공유)"
            )
        # ── [2026-05-26] catalog_ranker (LLM batch ranking) ──────────────
        # 후보 표 N개를 한 번에 LLM에 보내 의미 매칭 점수로 ranking.
        # 키워드 가드 + per-table relevance_judge를 통합 대체.
        # 비활성 시 (config or LLM 실패) 기존 키워드 가드로 fallback.
        _ranker_cfg = (
            ((context.config or {}).get("data_sources") or {})
            .get("kosis") or {}
        ).get("catalog_ranker") or {}
        _ranker_enabled = bool(_ranker_cfg.get("enabled", False))

        # 후보 pool 구성 — ranker 활성/비활성에 따라 다름
        if _ranker_enabled and context.workspace is not None:
            # 메타데이터 풍부한 pool 수집 (catalog + explore union)
            _pool_limit = int(_ranker_cfg.get("pool_limit", 20))
            _pool = _collect_candidate_pool(
                context.workspace, context.claim_id, candidate_id,
                limit=_pool_limit,
            )
            # current_id가 _pool에 없으면 (현재 pool은 current_id를 seen으로 skip)
            # candidate 정보 빠지므로 catalog observation에서 보강
            _current_cand: dict | None = None
            try:
                for _obs_name in context.workspace.list_observations(context.claim_id):
                    if "catalog_search" not in _obs_name.lower():
                        continue
                    _obs = context.workspace.read_observation(context.claim_id, _obs_name)
                    if not isinstance(_obs, dict):
                        continue
                    for _c in (_obs.get("output") or {}).get("candidates") or []:
                        if isinstance(_c, dict) and _c.get("id") == candidate_id:
                            _current_cand = dict(_c)
                            _current_cand["_pool_source"] = "catalog"
                            break
                    if _current_cand:
                        break
            except Exception:
                pass
            if _current_cand is None:
                _current_cand = {"id": candidate_id, "name": "", "score": 0.0, "_pool_source": "catalog"}

            # 전체 ranking 대상: current + prior_success + pool. 중복 제거.
            _rank_input: list[dict] = []
            _rank_seen: set[str] = set()
            def _add_to_rank(cand: dict) -> None:
                cid = str(cand.get("id") or "")
                if not cid or cid in _rank_seen:
                    return
                _rank_seen.add(cid)
                _rank_input.append(cand)

            _add_to_rank(_current_cand)
            for sid in prior_success_ids:
                if sid in _rank_seen:
                    continue
                _add_to_rank({"id": sid, "name": "[prior_success]", "score": 0.0, "_pool_source": "prior_success"})
            for c in _pool:
                _add_to_rank(c)

            # ranker 호출
            from structverify.retrieval.catalog_ranker import rank_candidates
            _ranker_threshold = float(_ranker_cfg.get("score_threshold", 0.15))
            try:
                _rankings = await rank_candidates(
                    claim_text=str(params.get("raw_claim") or params.get("claim_text") or ""),
                    indicator=str(params.get("indicator") or ""),
                    population=str(params.get("population") or ""),
                    time_period=str(params.get("time_period") or ""),
                    parent_path=str(params.get("parent_path") or ""),
                    candidates=_rank_input,
                    config=context.config,
                )
            except Exception as _e:
                logger.warning(f"[fetch_evidence] catalog_ranker 호출 예외: {_e}")
                _rankings = None

            if _rankings:
                # prior_success는 별도 캐시 가치라 ranker 점수 외에 *최우선 유지*.
                _prior_set = set(prior_success_ids)
                _ranked_ids = [
                    r["id"] for r in _rankings
                    if r["score"] >= _ranker_threshold and r["id"] not in _prior_set
                ]
                _rejected_ids = [
                    r["id"] for r in _rankings if r["score"] < _ranker_threshold
                ]
                # try_ids: prior_success(최우선) → ranker top → 거부된 표 (안전망, 마지막 시도)
                try_ids = []
                for sid in prior_success_ids + _ranked_ids:
                    if sid and sid not in try_ids:
                        try_ids.append(sid)
                # current candidate_id가 reject 됐어도 *맨 뒤*에 한 번 더 시도 (안전망)
                if candidate_id and candidate_id not in try_ids:
                    try_ids.append(candidate_id)
                _max_try = int(_ranker_cfg.get("max_try", 10))
                try_ids = try_ids[:_max_try]
                # ── ranker 결정 로그: before/after 명시 + 각 표의 score+reason ──
                _before_ids = [c["id"] for c in _rank_input if c.get("id")]
                logger.info(
                    f"[fetch_evidence] catalog_ranker decision:\n"
                    f"  input ({len(_before_ids)}): {_before_ids}\n"
                    f"  output try_ids (top {len(try_ids)}): {try_ids}\n"
                    f"  rejected<{_ranker_threshold}: {_rejected_ids[:5]}"
                    f"{'...' if len(_rejected_ids) > 5 else ''}"
                )
                for r in _rankings[:10]:
                    _mark = (
                        "★" if r["score"] >= _ranker_threshold and r["id"] in try_ids[:3]
                        else " "
                    )
                    logger.info(
                        f"  {_mark} rank: id={r['id']} score={r['score']:.2f} "
                        f"reason={r.get('reason', '')[:140]!r}"
                    )
            else:
                # ranker 실패 — fallback: 기존 candidate_id + prior + fallback ids
                logger.warning("[fetch_evidence] catalog_ranker 미적용 (실패) — 기본 순서 사용")
                try_ids = []
                for sid in [candidate_id] + prior_success_ids + list(fallback_ids):
                    if sid and sid not in try_ids:
                        try_ids.append(sid)
                try_ids = try_ids[:5]
        else:
            # ── 기존 동작 (ranker 비활성) ─────────────────────────────
            # try_ids: top → prior_success → catalog fallback. 중복 제거.
            try_ids = []
            for sid in [candidate_id] + prior_success_ids + list(fallback_ids):
                if sid and sid not in try_ids:
                    try_ids.append(sid)
            try_ids = try_ids[:5]  # 상한 5개 유지

            # Indicator Semantic Guard (키워드 룰 fallback)
            try:
                _indicator_str = str(params.get("indicator") or "")
                _SPECIFIC_KWS = (
                    "체외", "쇄석", "충격파",
                    "진단방사선", "특수의료", "특수의 료",
                    "엑스선", "X선", "엑스레이",
                    "CT", "MRI", "PET", "초음파",
                    "방사선", "단층",
                )
                _claim_has_specific = any(
                    kw.lower() in _indicator_str.lower() for kw in _SPECIFIC_KWS
                )

                _id_to_name: dict[str, str] = {}
                if context.workspace is not None:
                    try:
                        for _obs_name in context.workspace.list_observations(context.claim_id):
                            if "catalog_search" not in _obs_name.lower():
                                continue
                            _obs = context.workspace.read_observation(context.claim_id, _obs_name)
                            if not isinstance(_obs, dict):
                                continue
                            for _c in (_obs.get("output") or {}).get("candidates") or []:
                                if isinstance(_c, dict):
                                    _cid, _cname = _c.get("id"), _c.get("name")
                                    if _cid and _cname and _cid not in _id_to_name:
                                        _id_to_name[_cid] = _cname
                    except Exception:
                        pass

                def _name_specificity(name: str) -> bool:
                    if not name:
                        return False
                    return any(kw.lower() in name.lower() for kw in _SPECIFIC_KWS)

                def _semantic_score(sid: str) -> int:
                    if sid in prior_success_ids:
                        return 3
                    _name = _id_to_name.get(sid, "")
                    _name_has_specific = _name_specificity(_name)
                    if _claim_has_specific and _name_has_specific:
                        return 2
                    if not _claim_has_specific and not _name_has_specific:
                        return 1
                    if _claim_has_specific and not _name_has_specific:
                        return 0
                    return -1

                _before = list(try_ids)
                try_ids.sort(key=lambda s: -_semantic_score(s))
                if try_ids != _before:
                    logger.info(
                        f"[fetch_evidence] indicator semantic guard reorder: "
                        f"indicator={_indicator_str!r} "
                        f"(claim_has_specific={_claim_has_specific}) "
                        f"{_before} → {try_ids}"
                    )
            except Exception as _e:
                logger.debug(f"[fetch_evidence] indicator semantic guard 실패 (무시): {_e}")

        evidence = None
        used_id = candidate_id
        last_err: str | None = None
        # [패치 E] prior_success_ids로 들어온 stat_id는 표 이름 기반 관련성
        # 가드를 우회하기 위해 params에 플래그를 단다. 같은 job에서 이미
        # 한 번 fetch 성공한 표는 row data 안에 indicator가 있을 가능성이
        # 입증된 것이므로, 표 이름이 indicator와 안 닿더라도 일단 fetch
        # 시도해서 _select_best_row가 진짜 row를 찾게 한다.
        prior_id_set = set(prior_success_ids)
        # [2026-05-26] fetched_values 캐시 — fetch 진입 전 lookup용
        _ws_for_cache = getattr(context, "workspace", None)
        _ind_for_cache = str(params.get("indicator") or "")
        _tp_for_cache = str(params.get("time_period") or "")
        _pop_for_cache = str(params.get("population") or "")
        for idx, try_id in enumerate(try_ids):
            # ── [2026-05-26] fetched_values 캐시 lookup ────────────────
            # 같은 (stat_id, indicator, time, population) 조합이 이미 fetch
            # 됐으면 source.fetch_evidence 호출 안 하고 캐시 값 반환.
            # 같은 claim 내 반복 fetch (LLM이 동일 data 재요청) 또는 다른
            # claim의 재사용 모두 처리.
            if _ws_for_cache is not None and _ind_for_cache and _tp_for_cache:
                try:
                    _cached_ev = _ws_for_cache.lookup_fetched_value(
                        stat_id=try_id,
                        indicator=_ind_for_cache,
                        time_period=_tp_for_cache,
                        population=_pop_for_cache,
                    )
                except Exception as _e:
                    logger.debug(f"[fetch_evidence] fetched_value lookup 실패: {_e}")
                    _cached_ev = None
                if _cached_ev is not None:
                    evidence = _cached_ev
                    used_id = try_id
                    logger.info(
                        f"[fetch_evidence] fetched_values 캐시 적중: "
                        f"stat_id={try_id} indicator={_ind_for_cache!r} "
                        f"time={_tp_for_cache!r} population={_pop_for_cache!r} "
                        f"value={_cached_ev.get('value') if hasattr(_cached_ev, 'get') else None} "
                        f"— source.fetch_evidence skip"
                    )
                    break

            try:
                call_params = dict(params)
                if try_id in prior_id_set:
                    call_params["_from_prior_success"] = True
                # [P20 2026-05-22] workspace 전달 — KOSIS raw 응답 캐시 활용
                ev = await source.fetch_evidence(
                    candidate_id=try_id, params=call_params,
                    workspace=_ws_for_cache,
                )
            except Exception as e:
                logger.warning(
                    f"[fetch_evidence] 후보 {try_id} fetch 예외: "
                    f"{type(e).__name__}: {e}"
                )
                last_err = f"{type(e).__name__}: {e}"
                continue
            # [2026-05-21 P6] ev dict이지만 value=None이면 *실패로 간주*하고 다음 후보 시도.
            # 기존엔 `ev is not None`만 봤어서 INH_1B83A35처럼 dict는 받았지만 row 비어
            # value=None인 케이스에서도 break해버려 다음 후보(DT_1B8000G)를 안 돌렸음.
            # → 단일 fetch로 끝나고 loop이 즉시 unverifiable로 떨어지는 버그.
            _ev_dict = dict(ev) if (ev is not None and hasattr(ev, "items")) else {}
            _ev_value = _ev_dict.get("value") if _ev_dict else None
            if ev is not None and _ev_value is not None:
                evidence = ev
                used_id = try_id
                # ── [2026-05-26] fetched_values 캐시 저장 ──────────────
                if _ws_for_cache is not None and _ind_for_cache and _tp_for_cache:
                    try:
                        _ws_for_cache.append_fetched_value(
                            stat_id=try_id,
                            indicator=_ind_for_cache,
                            time_period=_tp_for_cache,
                            population=_pop_for_cache,
                            evidence=_ev_dict,
                        )
                    except Exception as _e:
                        logger.debug(f"[fetch_evidence] fetched_value 저장 실패: {_e}")
                if idx > 0:
                    logger.info(
                        f"[fetch_evidence] top 후보 실패 → 후보 {idx+1}번째 "
                        f"{try_id} 로 성공"
                    )
                break
            else:
                _why = "None" if ev is None else "value=None"
                logger.info(
                    f"[fetch_evidence] 후보 {try_id} → {_why} "
                    f"(관련성 거부/데이터 없음), 다음 후보 시도"
                )
                # [P33b 2026-05-22] 실패한 stat_id를 workspace blacklist에 기록
                # → 다음 catalog_search에서 결과에서 제외 → 같은 표 무한 반복 차단.
                try:
                    _ws = getattr(context, "workspace", None)
                    if _ws is not None and hasattr(_ws, "append_failed_stat_id"):
                        _ws.append_failed_stat_id(
                            context.claim_id, try_id, reason=f"fetch_failed_{_why}",
                        )
                except Exception as _e:
                    logger.debug(f"[fetch_evidence] failed_stat_id 기록 실패: {_e}")
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

        # [패치 A] job-level successful stat_id 저장 (다음 claim이 이 표를 1순위로)
        try:
            sid_for_save = evidence_dict.get("stat_table_id") or candidate_id
            context.workspace.append_successful_stat_id(str(sid_for_save))
        except Exception as e:
            logger.debug(f"[fetch_evidence] successful_stat_id 저장 실패: {e}")

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

        # ── [T 패치 2026-05-21] sibling_evidence에 직접 저장 ──
        # 기존엔 _save_verified_facts (verdict가 match/mismatch일 때만) 경유 →
        # verdict가 unverifiable이거나 data_points 비어있으면 sibling 저장 누락.
        # fetch가 *성공*한 시점에 *KOSIS 값을 받아온 사실 자체*가 sibling이 활용할
        # 신호이므로 verdict 무관하게 여기서 직접 저장 (verdict='fetched' 표시).
        try:
            if value is not None and claim is not None:
                _schema = getattr(claim, "schema", None)
                _sent_id = str(getattr(claim, "sent_id", "") or "").strip()
                _role = (getattr(_schema, "value_role", None) or "") if _schema else ""
                if _sent_id and _role and hasattr(context.workspace, "record_sibling_evidence"):
                    context.workspace.record_sibling_evidence(
                        sent_id=_sent_id,
                        role=_role,
                        evidence={
                            "indicator": params.get("indicator") or "",
                            "value": value,
                            "unit": unit or "",
                            "time_period": time_period or params.get("time_period") or "",
                            "source": (
                                f"kosis:{evidence_dict.get('stat_table_id') or candidate_id}"
                            ),
                            "claim_id": str(context.claim_id),
                            "verdict": "fetched",
                        },
                    )
        except Exception as e:
            logger.debug(f"[fetch_evidence] sibling_evidence 저장 실패 (무시): {e}")

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