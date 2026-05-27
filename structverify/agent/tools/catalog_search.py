"""
structverify.agent.tools.catalog_search — 카탈로그 검색 Tool.

Agent가 *키워드 → 데이터 표/지표 후보*를 찾을 때 호출.

작동:
  1. context.datasources에서 source 선택 (기본: default_source)
  2. source.search_catalog(query, category, top_k) 호출
  3. 후보 리스트 반환 + workspace observation 저장

source는 BaseDataSource 인터페이스를 구현한 *어떤 것이든* 사용 가능 (KOSIS, custom_csv, ...).
Phase B에서는 *추상 인터페이스만* — 실제 KOSIS DataSource 구현은 Phase D에서 wiring.

호출 예:
  input = {"query": "출생아 수 4월", "category": ["인구", "출생"], "top_k": 5}
  → 5개 후보 candidates: [{id, name, score, ...}, ...]
"""
from __future__ import annotations

import json
from typing import Any
from structverify.utils.logger import get_logger

from ..schemas import ActionType
from .base import ToolBase, ToolContext, ToolResult, register_tool

logger = get_logger(__name__)


def _read_last_explore_categories(workspace, claim_id, top_n: int = 2) -> list[str]:
    """[R1.5] 같은 claim의 직전 explore_catalog observation에서 top N 카테고리 추출.

    explore_catalog가 만든 observation은 name이 'iter{NNN}_explore_catalog' 형식.
    가장 최근(iter 큰) 것을 골라서 categories[].category_label top_n개 반환.
    LLM이 catalog_search에 category를 안 넘기거나 KOSIS 어휘와 안 맞는 자유어를
    넘겨도, 시스템이 임베딩으로 찾은 정확한 KOSIS 카테고리를 보강할 수 있도록.
    """
    if workspace is None or not claim_id:
        return []
    try:
        names = workspace.list_observations(claim_id)
    except Exception:
        return []
    # iter 큰(가장 최근) explore observation 우선
    explore_names = sorted(
        [n for n in names if "explore_catalog" in n.lower()],
        reverse=True,
    )
    if not explore_names:
        return []
    for name in explore_names:
        data = workspace.read_observation(claim_id, name)
        if not isinstance(data, dict):
            continue
        cats = (data.get("output") or {}).get("categories") or []
        labels: list[str] = []
        for c in cats[:top_n]:
            if not isinstance(c, dict):
                continue
            lbl = c.get("category_label")
            if lbl:
                labels.append(str(lbl))
        if labels:
            return labels
    return []


@register_tool(ActionType.CATALOG_SEARCH)
class CatalogSearchTool(ToolBase):
    """데이터 소스 카탈로그(표 목록) 검색.

    여러 source 중 *config.data_sources.default_source* (또는 input.source 명시) 사용.
    DataSource 추상화 덕분에 *KOSIS 외 회사 자체 DB/CSV*도 동일 인터페이스로 검색.
    """

    name = ActionType.CATALOG_SEARCH
    description = (
        "데이터 소스의 표/지표 카탈로그를 키워드로 검색. "
        "한 번 시도해서 잘못된 표가 나오면 *다른 검색어*로 재시도 권장. "
        "memory에 *이미 시도한 검색어* 있으면 중복 금지."
    )
    input_schema = {
        "query": "검색 키워드 (한국어, 3-6단어 권장). 예: '출생아 수 인구동향'",
        "category": "(선택) 분류 힌트 리스트. 예: ['인구', '출생']",
        "top_k": "(선택) 최대 후보 수. 기본 5",
        "source": "(선택) 데이터 소스 이름. 기본은 config.data_sources.default_source",
        "force_explore": (
            "(선택) true면 catalog top1 점수와 무관하게 deep_explore "
            "(top 표들에서 sample row 가져와 LLM이 row 단서 기반 reasoning) "
            "강제 발동. fetch_evidence가 row를 못 찾아 회복이 필요할 때 사용."
        ),
        "explore_mode": (
            "(선택) force_explore=true 시 사용할 방식. 'meta'(기본, 권장): KOSIS "
            "getMeta(ITM/OBJ) 호출로 표 항목/분류 list만 받아 LLM이 정답 표 식별 "
            "(빠름, ~표당 1초). 'row_preview': 표 전체 row preview 받아 LLM이 외삽 "
            "reasoning (느림, 표당 22초). 'meta'가 정답 표 직접 식별에 유리."
        ),
        "query_rewrite": (
            "(선택) true면 catalog_search 실행 *전에* LLM이 query를 표 이름 친화 "
            "어휘로 변형(예: '체외 충격파 쇄석술 장비 수' → '시군구별 의료장비'). "
            "원본 query에 row-level keyword만 있어 catalog 후보에 정답 표가 안 들어올 "
            "때 사용. 각 변형으로 search → 합집합 반환."
        ),
    }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        query = (input_data.get("query") or "").strip()
        if not query:
            return ToolResult(
                output={},
                summary="실패: query 비어있음",
                success=False,
                error="query는 비어있을 수 없습니다.",
            )

        category = input_data.get("category") or None
        if category is not None and not isinstance(category, list):
            category = [str(category)]

        try:
            top_k = int(input_data.get("top_k") or 15)
        except (TypeError, ValueError):
            top_k = 10
        # [P29' 2026-05-22] 기본 5 → 15. catalog 임베딩이 표 이름만 보는 한계로
        # row-level keyword 쿼리(예: "체외 충격파 쇄석술 장비")는 정답 표가
        # cosine 6~15위에 깔리는 경우가 있어 top 5만 노출하면 reflect/deep_explore
        # 모두 정답에 접근 불가. 19개 합집합을 거의 그대로 노출해도 prompt
        # context 부담은 적당하고 정답 진입률 ↑.
        top_k = max(15, min(top_k, 20))

        # ── [패치 R1.5] explore_catalog 결과 자동 활용 ─────────────────
        # LLM이 explore_catalog 결과를 무시하고 자기 머릿속 카테고리 어휘를
        # 그대로 catalog_search에 넘기는 케이스가 잦다 (예: ['기후 변화', '날씨 정보']).
        # 직전 explore observation을 자동 읽어, LLM 카테고리에 explore가 추천한
        # top 2 카테고리를 union으로 추가. 임베딩 의미 검색이 찾아준 정확한
        # KOSIS 카테고리 어휘 (예: '기상관측통계')가 ILIKE 필터에 들어가
        # 무관한 카테고리만 봐서 헛돌이가 되는 걸 방지.
        try:
            explored_cats = _read_last_explore_categories(
                context.workspace, context.claim_id, top_n=2,
            )
        except Exception as e:
            logger.debug(f"[catalog_search] explore observation 읽기 실패: {e}")
            explored_cats = []
        if explored_cats:
            existing = set(category or [])
            new_cats = [c for c in explored_cats if c and c not in existing]
            if new_cats:
                category = list(category or []) + new_cats
                logger.info(
                    f"[catalog_search] 직전 explore_catalog top 카테고리 자동 추가: "
                    f"{new_cats} → 최종 category={category}"
                )

        # source 선택
        ds_config = context.config.get("data_sources", {}) if context.config else {}
        default_source = ds_config.get("default_source", "kosis")
        source_name = (input_data.get("source") or default_source).strip()

        # DataSource 인스턴스 찾기
        source = context.datasources.get(source_name) if context.datasources else None
        if source is None:
            available = list(context.datasources.keys()) if context.datasources else []
            return ToolResult(
                output={"requested_source": source_name, "available": available},
                summary=f"실패: source={source_name!r} 등록 안 됨. 가능: {available}",
                success=False,
                error=(
                    f"DataSource '{source_name}'이 context.datasources에 없습니다. "
                    f"가능한 source: {available}. config.data_sources.enabled 확인."
                ),
            )

        # ── [P30 2026-05-22] query_rewrite — 검색 *전*에 LLM이 query 변형 ──
        # query가 row-level keyword면 catalog 표 이름과 매칭이 약함. LLM이
        # 표 이름 친화 어휘로 변형 후 각 변형으로 검색 → 합집합. 원본 + 변형
        # 둘 다 시도해 정답 표 진입률 ↑.
        _queries_to_search: list[str] = [query]
        _qr_applied: list[str] = []
        if bool(input_data.get("query_rewrite")) and context.claim is not None:
            try:
                from .query_rewriter import rewrite_query as _rewrite_query
                _variations = await _rewrite_query(
                    query=query, claim=context.claim, config=context.config,
                )
                for v in _variations:
                    if v and v not in _queries_to_search:
                        _queries_to_search.append(v)
                        _qr_applied.append(v)
                if _qr_applied:
                    logger.info(
                        f"[catalog_search] query_rewrite 적용 — 원본={query!r} + "
                        f"변형 {len(_qr_applied)}개: {_qr_applied}"
                    )
            except Exception as _e:
                logger.warning(f"[catalog_search] query_rewrite 실패: {_e}")

        # [P31 2026-05-22] claim에서 schema 정보 추출 → source.search_catalog의
        # context로 전달. KOSIS DataSource는 이걸 ConnectorQuery.extra_params에 묶어
        # _extract_category_and_keyword에 전달하므로:
        #   - parent_path 있으면 LLM 카테고리 추출 *skip*하고 KOSIS 어휘 그대로 사용
        #   - raw_claim 있으면 LLM이 전체 문장 보고 더 정확한 카테고리/검색어
        # claim/schema가 없으면 None인 채로 안전하게 동작.
        _ctx_for_source: dict[str, Any] = {}
        _claim = context.claim
        if _claim is not None:
            _claim_text = getattr(_claim, "claim_text", None)
            if _claim_text:
                _ctx_for_source["raw_claim"] = str(_claim_text)[:200]
            _schema = getattr(_claim, "schema", None)
            if _schema is not None:
                # [2026-05-27 Fix B] time_period 포함 — catalog_search에서 시점 토큰
                # 기반 union 검색을 발동시키기 위해 schema.time_period를 context에
                # 실어 보낸다. kosis_source._make_query가 ConnectorQuery.time_period로
                # 매핑하고 CatalogSearch가 "embedding_text + year"로 추가 pgvector
                # 검색을 돌려 dedup union.
                for _k in ("parent_path", "population", "indicator", "time_period"):
                    _v = getattr(_schema, _k, None)
                    if _v:
                        _ctx_for_source[_k] = str(_v)

        # 검색 실행 — 단일 query 또는 다중 query 합집합
        try:
            if len(_queries_to_search) == 1:
                candidates = await source.search_catalog(
                    query=query, category=category, top_k=top_k,
                    context=_ctx_for_source or None,
                )
            else:
                import asyncio as _aio
                _all = await _aio.gather(
                    *[
                        source.search_catalog(
                            query=q, category=category, top_k=top_k,
                            context=_ctx_for_source or None,
                        )
                        for q in _queries_to_search
                    ],
                    return_exceptions=True,
                )
                # 합집합 + 중복 제거 (id 기준)
                candidates = []
                _seen: set[str] = set()
                for res in _all:
                    if isinstance(res, Exception):
                        continue
                    for c in (res or []):
                        cid = c.get("id") if hasattr(c, "get") else None
                        if not cid or cid in _seen:
                            continue
                        _seen.add(cid)
                        candidates.append(c)
                # 점수순 정렬 (각 query의 score를 그대로 사용)
                candidates.sort(
                    key=lambda c: float(c.get("score", 0.0) or 0.0),
                    reverse=True,
                )
        except Exception as e:
            logger.exception(f"[catalog_search] source={source_name} query={query!r} 실패")
            return ToolResult(
                output={"source": source_name, "query": query},
                summary=f"실패: catalog_search({source_name}) — {type(e).__name__}: {e}",
                success=False,
                error=f"{type(e).__name__}: {e}",
            )

        # 결과 정규화 (CatalogCandidate dict 호환)
        normalized: list[dict[str, Any]] = []
        for c in candidates or []:
            # dict이거나 dict-like
            d = dict(c) if hasattr(c, "items") else {}
            normalized.append(d)

        # ── [P33b 2026-05-22] 이전 fetch 실패 stat_id 제외 ────────────
        # 같은 claim에서 catalog_search → fetch_evidence가 *관련성 거부* 또는
        # *row 매칭 실패*로 None 반환한 표는 *다음 catalog_search에서도 다시
        # 후보로 올라옴* (KOSIS 검색이 deterministic). reflect가 thought엔
        # "다른 표 시도"라 쓰지만 input은 거의 같은 query라 결국 같은 5개
        # 후보 → 같은 표 반복 거부 → 무한 헛돌이.
        # workspace에 *이 claim에서 실패한 stat_id 목록*을 두고 결과에서 제거.
        _failed_count = 0
        try:
            _failed_ids = set(
                context.workspace.read_failed_stat_ids(context.claim_id)
                if context.workspace else []
            )
        except Exception:
            _failed_ids = set()
        if _failed_ids:
            _before = len(normalized)
            normalized = [c for c in normalized if c.get("id") not in _failed_ids]
            _failed_count = _before - len(normalized)
            if _failed_count > 0:
                logger.info(
                    f"[catalog_search] 이전 fetch 실패 stat_id {_failed_count}개 제외 "
                    f"({sorted(_failed_ids)[:5]}{'...' if len(_failed_ids) > 5 else ''}) "
                    f"— 같은 표 무한 반복 방지"
                )

        # ── [패치 D] job에서 이미 fetch 성공한 stat_id를 결과 맨 앞에 prepend ──
        # 같은 KOSIS 표가 여러 지표(출생아 수/합계출산율/혼인 건수)를 같이
        # 갖고 있는데 catalog는 검색어별로 다른 표를 top으로 주는 경우 대응.
        # 예: '합계출산율' 검색 시 catalog top은 'DT_XNN0004(해외)'인데
        # 같은 job의 다른 claim이 이미 'DT_1B8000G(국내 인구동향)'에서
        # 성공했고 그 표 안에 합계출산율 row가 있으므로 이걸 1순위로
        # 노출시켜 reflect agent가 fetch_evidence를 호출하게 유도.
        try:
            prior_ids = context.workspace.read_successful_stat_ids() if context.workspace else []
        except Exception:
            prior_ids = []
        if prior_ids:
            existing_ids = {c.get("id") for c in normalized if c.get("id")}
            # [패치 D'] prepend 시 진짜 stat_name도 같이 보여줘서 LLM이
            # "이 표 안에 다른 지표도 있을 가능성"을 판단할 수 있게 한다.
            # workspace에 저장한 catalog observation에서 stat_id의 name을 찾아옴.
            # [2026-05-21 P5] 이전 observation의 name엔 우리가 박아둔 "[...]"
            # annotation이 누적될 수 있음 — 새로 prepend할 때 그걸 떼서 *원본 표 이름*만
            # 가져오도록 정규화한다. 안 그러면 매 검색마다 "[표 안에 'X' 관련 ...]
            # [표 안에 'Y' 관련 ...]" 식으로 거짓 힌트가 무한 누적된다.
            import re as _re_local
            _ANN_RE = _re_local.compile(r"\s*\[같은 job에서[^\]]*\]\s*")
            def _clean_name(nm: str) -> str:
                return _ANN_RE.sub("", nm or "").strip()
            stat_names: dict[str, str] = {}
            try:
                for other_cid in context.workspace.list_claims():
                    for obs_name in context.workspace.list_observations(other_cid):
                        if "catalog_search" not in obs_name.lower():
                            continue
                        data = context.workspace.read_observation(other_cid, obs_name)
                        if not isinstance(data, dict):
                            continue
                        for cd in (data.get("output") or {}).get("candidates") or []:
                            sid = cd.get("id") if isinstance(cd, dict) else None
                            nm = cd.get("name") if isinstance(cd, dict) else None
                            if sid and nm and sid not in stat_names:
                                stat_names[sid] = _clean_name(nm)
                    if len(stat_names) > 50:
                        break  # 충분히 많이 모음
            except Exception as e:
                logger.debug(f"[catalog_search] stat_name lookup 실패: {e}")

            # [2026-05-21 P5] 각 prior stat_id가 *어떤 지표* 검증에 사용됐는지
            # verified_facts에서 역추적. source 필드는 보통 "KOSIS:DT_..." 또는
            # "kosis:DT_..." 형식이라 case-insensitive substring 매칭. 이게 있으면
            # LLM이 "이 표는 출생아 수 검증에 썼던 표 — 합계출산율이랑 같은 인구동향
            # 표일 가능성"을 판단할 수 있다. 거짓 "row 있을 가능성" 단언은 제거.
            stat_id_to_indicators: dict[str, list[str]] = {}
            try:
                _facts = context.workspace.read_verified_facts() if context.workspace else []
            except Exception:
                _facts = []
            for _f in _facts or []:
                _src = str(_f.get("source") or "").lower()
                _ind = str(_f.get("indicator") or "").strip()
                if not _ind:
                    continue
                for sid in prior_ids:
                    if sid.lower() in _src:
                        bucket = stat_id_to_indicators.setdefault(sid, [])
                        if _ind not in bucket:
                            bucket.append(_ind)
            # sibling_evidence의 source는 "kosis:{stat_id}" 형식 — 같이 합쳐 정확도 ↑
            try:
                if hasattr(context.workspace, "_sibling_evidence_key"):
                    # 모든 sent_id를 모르므로 backend로 직접 읽기
                    _key = context.workspace._sibling_evidence_key()
                    if context.workspace.backend.exists(_key):
                        _sib_data = json.loads(
                            context.workspace.backend.read_text(_key)
                        )
                        if isinstance(_sib_data, dict):
                            for _sent_id, _entries in _sib_data.items():
                                if not isinstance(_entries, list):
                                    continue
                                for _e in _entries:
                                    if not isinstance(_e, dict):
                                        continue
                                    _src = str(_e.get("source") or "").lower()
                                    _ind = str(_e.get("indicator") or "").strip()
                                    if not _ind:
                                        continue
                                    for sid in prior_ids:
                                        if sid.lower() in _src:
                                            bucket = stat_id_to_indicators.setdefault(sid, [])
                                            if _ind not in bucket:
                                                bucket.append(_ind)
            except Exception as e:
                logger.debug(f"[catalog_search] sibling_evidence 읽기 실패 (무시): {e}")

            # [P21 2026-05-22] prior_success score 완화 — 1.5 고정 X.
            # 기존 1.5 박으면 *의미적으로 더 잘 맞는 catalog 결과 (score=1.0)*가
            # 무조건 prior 표 뒤로 밀림. "치료 가능 사망률" claim에 의료장비 표
            # (DT_35003_A7 prior)가 top이 되고 진짜 정답(DT_117049_A083 score=1.0)이
            # 3순위로 깔리는 회귀 케이스 (22:54 로그).
            # → prior bonus는 catalog top score 살짝 *아래*로. catalog 매칭이
            # 명확하면(top score ≥ 0.9) prior는 *후순위*로 보이고, catalog가 약하면
            # (top score < prior_bonus) prior가 자연스럽게 위로 올라옴.
            _catalog_top_score = max(
                (float(c.get("score", 0.0) or 0.0) for c in normalized),
                default=0.0,
            )
            # prior score: catalog top score - 0.05. 단 최소 0.5는 보장 (catalog 결과
            # 자체가 없거나 약할 때도 prior가 fetch 후보로 살아남도록).
            _prior_score = max(_catalog_top_score - 0.05, 0.5)
            prepend: list[dict[str, Any]] = []
            for sid in prior_ids:
                if sid in existing_ids:
                    continue
                real_name = stat_names.get(sid, sid)
                _prior_inds = stat_id_to_indicators.get(sid) or []
                if _prior_inds:
                    _ind_label = ", ".join(f"'{x}'" for x in _prior_inds[:3])
                    _hint = (
                        f"[같은 job에서 {_ind_label} 검증에 사용된 표 — "
                        f"'{query}' row 존재는 fetch 시도로 확인]"
                    )
                else:
                    # indicator 추적 못 한 경우 *중립적* 라벨 (거짓 단언 금지)
                    _hint = "[같은 job 다른 claim이 사용한 표 — 관련성 fetch로 확인]"
                prepend.append({
                    "id": sid,
                    "name": f"{real_name} {_hint}",
                    "score": _prior_score,
                    "raw": {"from_job_success": True},
                })
            if prepend:
                logger.info(
                    f"[catalog_search] job-success stat_id "
                    f"{[p['id'] for p in prepend]} prior 추가 "
                    f"(score={_prior_score:.3f}, catalog top={_catalog_top_score:.3f})"
                )
                # score 순으로 다시 정렬 — prior가 catalog top 위에 있을지 아래일지는
                # score 비교에 맡김.
                normalized = sorted(
                    prepend + normalized,
                    key=lambda c: float(c.get("score", 0.0) or 0.0),
                    reverse=True,
                )

        # ── [P28 2026-05-22] Deep Exploration (T1 — 사전 보강) ────────────
        # catalog top1 점수가 낮거나 top1-top2 gap이 작으면, top N 표의 sample row를
        # 가져와 LLM이 row 단서 기반 reasoning으로 best 표 추천. P21B와 달리:
        #   - top 3 (기본), 표당 row 5개, Y prdSe만 → KOSIS 부하 ↓
        #   - prompt: "row 단서로 더 파볼 가치 있는 표 외삽 추천" (best 단순 선택 X)
        #   - none_signal → output에 표시 → reflect가 query refinement 결정
        # 또 force_explore=True (T2 회복용)면 점수 조건 무시하고 발동.
        _cs_cfg = (context.config or {}).get("catalog_search") or {}
        _dx_cfg = _cs_cfg.get("deep_explore") or {}
        _dx_enabled = bool(_dx_cfg.get("enabled", False))
        _force_explore = bool(input_data.get("force_explore"))
        _explore_meta: dict[str, Any] | None = None

        if _dx_enabled and normalized and context.claim is not None:
            _low_score_thr = float(_dx_cfg.get("trigger_low_score") or 0.6)
            _gap_thr = float(_dx_cfg.get("trigger_score_gap") or 0.1)
            _top1_score = float(normalized[0].get("score", 0.0) or 0.0)
            _top2_score = (
                float(normalized[1].get("score", 0.0) or 0.0)
                if len(normalized) >= 2 else 0.0
            )
            _t1_fire = (
                _top1_score < _low_score_thr
                or (_top1_score - _top2_score) < _gap_thr
            )
            # per-claim 발동 횟수 제한 (loop 방지)
            _max_per_claim = int(_dx_cfg.get("max_per_claim") or 2)
            _dx_obs_count = 0
            try:
                for _n in context.workspace.list_observations(context.claim_id):
                    if "deep_explore" in _n.lower():
                        _dx_obs_count += 1
            except Exception:
                pass

            if (_force_explore or _t1_fire) and _dx_obs_count < _max_per_claim:
                trigger_label = "T2/force" if _force_explore else "T1/score"
                # [P30 2026-05-22] explore_mode 분기:
                #   - "meta" (기본): getMeta(ITM/OBJ) → 빠름 + 정확
                #   - "row_preview": 표 전체 row preview → 느림 + 외삽
                # input의 explore_mode가 우선, 없으면 config의 default 사용.
                _explore_mode = str(
                    input_data.get("explore_mode")
                    or _dx_cfg.get("explore_mode")
                    or "meta"
                ).strip().lower()
                logger.info(
                    f"[catalog_search] deep_explore 발동 ({trigger_label}, "
                    f"mode={_explore_mode}) — top1={_top1_score:.3f}, "
                    f"gap={_top1_score-_top2_score:.3f}, prev_calls={_dx_obs_count}"
                )
                try:
                    if _explore_mode == "meta":
                        from .meta_explore import meta_explore as _explore_fn
                    else:
                        from .deep_explore import deep_explore as _explore_fn
                    _result = await _explore_fn(
                        query=query,
                        candidates=normalized,
                        claim=context.claim,
                        source=source,
                        workspace=context.workspace,
                        config=context.config,
                    )
                    if _result.used:
                        _explore_meta = {
                            "best_table_id": _result.best_table_id,
                            "reasoning": _result.reasoning,
                            "none_signal": _result.none_signal,
                            "previewed_ids": _result.previewed_ids,
                            "trigger": trigger_label,
                            "mode": _explore_mode,
                        }
                        # workspace observation으로도 별도 기록 (per-claim counter용)
                        try:
                            context.workspace.write_observation(
                                context.claim_id,
                                f"iter{context.iter_num:03d}_deep_explore",
                                _explore_meta,
                            )
                        except Exception as _e:
                            logger.debug(
                                f"[catalog_search] deep_explore observation 저장 실패: {_e}"
                            )
                        # best가 있으면 해당 표를 normalized top으로 승격
                        if _result.best_table_id and not _result.none_signal:
                            _ids = [c.get("id", "") for c in normalized]
                            if _result.best_table_id in _ids:
                                _idx = _ids.index(_result.best_table_id)
                                if _idx != 0:
                                    normalized = (
                                        [normalized[_idx]]
                                        + [c for i, c in enumerate(normalized) if i != _idx]
                                    )
                                    logger.info(
                                        f"[catalog_search] deep_explore best="
                                        f"{_result.best_table_id!r} → top1 승격"
                                    )
                except Exception as _e:
                    logger.warning(
                        f"[catalog_search] deep_explore 실패 (catalog 원본 유지): {_e}"
                    )
            elif _dx_obs_count >= _max_per_claim:
                logger.info(
                    f"[catalog_search] deep_explore skip — claim당 호출 한도 "
                    f"({_dx_obs_count}/{_max_per_claim})"
                )

        # ── [P21B 2026-05-22] Row-aware LLM rerank (P25에서 disable, P28과 별개) ───
        # top N 표에 sample row 1개 fetch해 LLM이 claim과 가장 잘 매칭되는 표 선택.
        # 임베딩 score만으론 "치료 가능 사망률"과 "사망률"·"의료장비"를 잘 못 구분
        # 하는 케이스 (23:54 로그) 대응. P20 KOSIS cache hit이면 비용 거의 0.
        _rerank_mode = str(_cs_cfg.get("rerank_mode") or "none").lower()
        _rerank_top_n = int(_cs_cfg.get("rerank_top_n") or 5)
        _rerank_gap = float(_cs_cfg.get("rerank_min_score_gap") or 0.15)
        if (
            _rerank_mode in ("row_preview", "table_name")
            and len(normalized) >= 2
            and context.claim is not None
        ):
            try:
                _rerank_top = normalized[:_rerank_top_n]
                _top1 = float(_rerank_top[0].get("score", 0.0) or 0.0)
                _top2 = float(_rerank_top[1].get("score", 0.0) or 0.0)
                if _top1 - _top2 >= _rerank_gap:
                    logger.info(
                        f"[catalog_search] rerank skip — top1({_top1:.3f}) - "
                        f"top2({_top2:.3f}) gap={_top1-_top2:.3f} ≥ {_rerank_gap} "
                        f"(LLM call 절약, top1이 이미 명확)"
                    )
                else:
                    _reranked = await _row_preview_rerank(
                        candidates=_rerank_top,
                        claim=context.claim,
                        source=source,
                        workspace=context.workspace,
                        config=context.config,
                        do_row_preview=(_rerank_mode == "row_preview"),
                    )
                    if _reranked:
                        # rerank된 top + 남은 후보 (원래 score 순서)
                        _reranked_ids = {c.get("id") for c in _reranked}
                        _rest = [c for c in normalized if c.get("id") not in _reranked_ids]
                        normalized = _reranked + _rest
                        logger.info(
                            f"[catalog_search] LLM rerank 적용 — "
                            f"top1={normalized[0].get('id')!r}"
                        )
            except Exception as _e:
                logger.warning(f"[catalog_search] rerank 실패 (catalog 원본 유지): {_e}")

        # workspace observation 저장 (raw)
        try:
            obs_name = f"iter{context.iter_num:03d}_catalog_search"
            context.workspace.write_observation(
                context.claim_id,
                obs_name,
                {"query": query, "category": category, "source": source_name,
                 "candidates": normalized},
            )
        except Exception as e:
            logger.debug(f"[catalog_search] observation 저장 실패: {e}")

        # 요약 (top 3 후보 이름)
        top_names = []
        for c in normalized[:3]:
            cid = c.get("id", "")
            cname = c.get("name", "")
            score = c.get("score")
            top_names.append(
                f"[{cid}] {cname}" + (f" (score={score:.3f})" if isinstance(score, (int, float)) else "")
            )
        summary_qr = (
            f" | query_rewrite +{len(_qr_applied)}" if _qr_applied else ""
        )
        summary = (
            f"catalog_search({source_name}) query={query!r} → "
            f"{len(normalized)}개 후보{summary_qr}. "
            f"Top: {' | '.join(top_names) if top_names else '(없음)'}"
        )

        _out: dict[str, Any] = {
            "source": source_name,
            "query": query,
            "category": category,
            "candidates": normalized,
            "candidate_count": len(normalized),
        }
        if _qr_applied:
            _out["_query_rewrite"] = {
                "original": query,
                "variations": _qr_applied,
            }
        if _explore_meta is not None:
            _out["_deep_explore"] = _explore_meta
            # summary에도 신호 추가 — reflect가 observation summary만 봐도 알도록
            if _explore_meta.get("none_signal"):
                summary += (
                    " | deep_explore: top 표들의 sample row에서 적합한 row 없음 — "
                    "다른 검색어로 catalog_search 재시도 권장."
                )
            elif _explore_meta.get("best_table_id"):
                summary += (
                    f" | deep_explore best={_explore_meta['best_table_id']} "
                    f"({_explore_meta.get('reasoning', '')[:60]})"
                )

        return ToolResult(
            output=_out,
            summary=summary,
            success=True,
        )


# ────────────────────────────────────────────────────────────────────
# [P21B 2026-05-22] Row-aware LLM rerank 헬퍼
# ────────────────────────────────────────────────────────────────────
import asyncio as _asyncio


async def _row_preview_rerank(
    candidates: list[dict[str, Any]],
    claim: Any,
    source: Any,
    workspace: Any,
    config: dict | None,
    do_row_preview: bool = True,
) -> list[dict[str, Any]] | None:
    """top N 후보에 대해 (옵션) sample row fetch + LLM rerank.

    Returns:
        rerank된 후보 list (best가 [0]). 실패/no-op이면 None.
    """
    if not candidates:
        return None

    # 1) (옵션) preview fetch — newEstPrdCnt=1로 KOSIS 1 row만 받아옴.
    #    P20 cache hit이면 즉시 반환. cold이면 KOSIS API 호출.
    previews: dict[str, dict] = {}
    if do_row_preview:
        async def _preview(cid: str) -> tuple[str, dict | None]:
            try:
                ev = await source.fetch_evidence(
                    candidate_id=cid,
                    params={
                        "newEstPrdCnt": "1",  # 가장 최근 1 시점만
                        "_preview": True,
                    },
                    workspace=workspace,
                )
                if ev is None:
                    return cid, None
                rows = ev.get("rows") or []
                first = rows[0] if rows else {}
                # 핵심 컬럼만 추출 (LLM 토큰 절약)
                _PICK = ("ITM_NM", "C1_NM", "C2_NM", "C3_NM", "C4_NM",
                          "PRD_DE", "UNIT_NM")
                row_summary = {k: first.get(k) for k in _PICK if first.get(k)}
                return cid, {
                    "stat_name": ev.get("stat_name") or "",
                    "sample_row": row_summary,
                    "rows_count": len(rows),
                }
            except Exception as _e:
                logger.debug(f"[catalog_search.rerank] preview fetch 실패 {cid}: {_e}")
                return cid, None

        results = await _asyncio.gather(
            *[_preview(c.get("id", "")) for c in candidates if c.get("id")],
            return_exceptions=False,
        )
        for cid, info in results:
            if info is not None:
                previews[cid] = info

    # 2) LLM 호출 — claim + 후보들 + preview를 던지고 best 1개 선택
    from structverify.utils.llm_client import LLMClient
    _llm = LLMClient(config=(config or {}).get("llm") or {})

    # claim 정보 추출
    _schema = getattr(claim, "schema", None)
    _info = {
        "indicator": (getattr(_schema, "indicator", None) or "") if _schema else "",
        "time_period": (getattr(_schema, "time_period", None) or "") if _schema else "",
        "population": (getattr(_schema, "population", None) or "") if _schema else "",
        "unit": (getattr(_schema, "unit", None) or "") if _schema else "",
        "value": (getattr(_schema, "value", None)) if _schema else None,
    }

    # 후보 리스트 → 프롬프트 텍스트
    _cand_lines: list[str] = []
    for i, c in enumerate(candidates, start=1):
        cid = c.get("id", "")
        cname = (c.get("name", "") or "").strip()
        cscore = c.get("score")
        _line = f"{i}. [{cid}] {cname}"
        if isinstance(cscore, (int, float)):
            _line += f" (catalog_score={cscore:.3f})"
        _preview = previews.get(cid)
        if _preview and _preview.get("sample_row"):
            _row = _preview["sample_row"]
            _row_str = ", ".join(f"{k}={v!r}" for k, v in _row.items())
            _line += f"\n   sample row → {_row_str}"
            _line += f" | rows_count={_preview.get('rows_count')}"
        _cand_lines.append(_line)

    _prompt = f"""다음 사실검증 claim에 *가장 적합한 KOSIS 통계표 1개*를 선택하세요.

[Claim]
- indicator (검증 대상 지표): {_info['indicator']!r}
- population (대상 집단): {_info['population']!r}
- time_period: {_info['time_period']!r}
- unit: {_info['unit']!r}
- value: {_info['value']}

[후보 표 (catalog 검색 score 순)]
{chr(10).join(_cand_lines)}

[선택 기준]
- ITM_NM 또는 C1_NM~C4_NM 어느 컬럼에 claim의 indicator가 *직접 포함*되거나 *의미적으로 매칭*되는 표 우선.
- C1_NM/C2_NM 등이 claim.population (지역/대상)을 포함하는 표 우선.
- 표 이름이 claim과 유사해 보이더라도 sample row의 ITM_NM이 무관하면 *배제*.
  (예: "치료 가능 사망률" claim에 sample row ITM_NM='의료장비'인 표는 배제)
- preview가 없는 표는 *catalog score만으로 추정* (불확실하다고 답해도 OK).

[응답 형식 — JSON only, 다른 텍스트 금지]
{{
  "best_stat_id": "DT_XXX",
  "reason": "한 줄 이유"
}}

선택을 못 하겠으면 best_stat_id를 첫 번째 후보 ID로.
"""

    try:
        raw = await _llm.generate(
            prompt=_prompt,
            system_prompt="KOSIS 통계표 선정 전문가. JSON으로만 답하세요.",
            model_tier="light",
        )
    except Exception as _e:
        logger.warning(f"[catalog_search.rerank] LLM 호출 실패: {_e}")
        return None

    # JSON 파싱 (관대하게)
    import json as _json
    import re as _re
    _best: str | None = None
    try:
        # raw에서 JSON 블록 추출
        _match = _re.search(r"\{[^{}]*\}", raw, _re.DOTALL)
        if _match:
            data = _json.loads(_match.group(0))
            _raw_best = (data.get("best_stat_id") or "").strip() or None
            # [P24 2026-05-22] LLM이 brackets/quotes 포함해서 반환하는 케이스 정규화.
            # prompt에 후보를 "1. [DT_XXX] 표이름" 형식으로 노출하니 LLM이 그대로
            # "[DT_XXX]" 복사하는 일이 잦음. brackets/quotes/공백 strip + (옵션)
            # candidates list에 substring 매칭 fallback.
            if _raw_best:
                _best = _raw_best.strip().strip("[]").strip("'\"").strip()
                if _best != _raw_best:
                    logger.info(
                        f"[catalog_search.rerank] best 정규화: {_raw_best!r} → {_best!r}"
                    )
            _reason = data.get("reason") or ""
            if _best:
                logger.info(
                    f"[catalog_search.rerank] LLM 선택: best={_best!r} "
                    f"reason={_reason[:80]!r}"
                )
    except Exception as _e:
        logger.debug(f"[catalog_search.rerank] LLM 응답 파싱 실패: {_e}")

    if not _best:
        return None

    # best를 0번으로 끌어올리기
    _ids = [c.get("id", "") for c in candidates]
    # [P24] substring fallback — 여전히 매칭 안 되면 candidates 중 best가 substring으로
    # 들어가는 표 찾기 (예: LLM이 'DT_117049_A083_2020' 대신 'DT_117049_A083' 반환).
    if _best not in _ids:
        _substring_match = None
        for _cid in _ids:
            if not _cid:
                continue
            if _best in _cid or _cid in _best:
                _substring_match = _cid
                break
        if _substring_match:
            logger.info(
                f"[catalog_search.rerank] best={_best!r} substring 매칭 → "
                f"{_substring_match!r} 사용"
            )
            _best = _substring_match
        else:
            logger.info(
                f"[catalog_search.rerank] best={_best!r}가 후보 list "
                f"{_ids[:5]}에 없음 — rerank skip"
            )
        return None
    _best_idx = _ids.index(_best)
    if _best_idx == 0:
        return None  # 이미 top — no-op
    _reordered = [candidates[_best_idx]] + [
        c for i, c in enumerate(candidates) if i != _best_idx
    ]
    return _reordered
