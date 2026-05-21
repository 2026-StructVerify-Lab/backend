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
            top_k = int(input_data.get("top_k") or 5)
        except (TypeError, ValueError):
            top_k = 5
        top_k = max(1, min(top_k, 20))

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

        # 검색 실행
        try:
            candidates = await source.search_catalog(
                query=query, category=category, top_k=top_k,
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
                                stat_names[sid] = nm
                    if len(stat_names) > 50:
                        break  # 충분히 많이 모음
            except Exception as e:
                logger.debug(f"[catalog_search] stat_name lookup 실패: {e}")
            prepend: list[dict[str, Any]] = []
            for sid in prior_ids:
                if sid in existing_ids:
                    continue
                real_name = stat_names.get(sid, sid)
                # 진짜 이름 + hint를 같이. LLM이 보고 fetch 시도하도록 유도.
                prepend.append({
                    "id": sid,
                    "name": (
                        f"{real_name} [같은 job에서 다른 claim이 이 표에서 데이터 "
                        f"가져왔음 — 표 안에 '{query}' 관련 row 있을 가능성]"
                    ),
                    "score": 1.5,  # 명시적 prior — top 위로 가도록 1.0 초과
                    "raw": {"from_job_success": True},
                })
            if prepend:
                logger.info(
                    f"[catalog_search] job-success stat_id {[p['id'] for p in prepend]} "
                    f"를 결과 맨 앞에 prepend (진짜 이름 포함)"
                )
                normalized = prepend + normalized

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
        summary = (
            f"catalog_search({source_name}) query={query!r} → "
            f"{len(normalized)}개 후보. Top: {' | '.join(top_names) if top_names else '(없음)'}"
        )

        return ToolResult(
            output={
                "source": source_name,
                "query": query,
                "category": category,
                "candidates": normalized,
                "candidate_count": len(normalized),
            },
            summary=summary,
            success=True,
        )
