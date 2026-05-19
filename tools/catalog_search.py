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

from structverify.utils.logger import get_logger
from typing import Any

from ..schemas import ActionType
from .base import ToolBase, ToolContext, ToolResult, register_tool

logger = get_logger(__name__)


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
