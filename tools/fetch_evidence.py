"""
structverify.agent.tools.fetch_evidence — 데이터 조회 Tool.

catalog_search로 *후보 발견* → fetch_evidence로 *실제 수치 조회*.

작동:
  1. context.datasources에서 source 선택 (catalog_search와 동일 source 권장)
  2. source.fetch_evidence(candidate_id, params) 호출
  3. EvidenceData 반환 + workspace observation 저장

source-specific 파라미터:
  - KOSIS: {"prdSe": "M", "startPrdDe": "202504", "endPrdDe": "202504", ...}
  - Custom CSV: {"row_filter": "month=4 AND year=2025", "column": "births"}
  - 외부 API: provider별 다름

Agent는 *params를 모르면 빈 dict {}로 호출*. DataSource가 *default 파라미터*로 fetch.
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
        "params는 source별 다름 (KOSIS는 시점 필터 등). 모르면 빈 dict {} 전달."
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

        # source 선택
        ds_config = context.config.get("data_sources", {}) if context.config else {}
        default_source = ds_config.get("default_source", "kosis")
        source_name = (input_data.get("source") or default_source).strip()

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

        # fetch 실행
        try:
            evidence = await source.fetch_evidence(
                candidate_id=candidate_id, params=params,
            )
        except Exception as e:
            logger.exception(
                f"[fetch_evidence] source={source_name} id={candidate_id} 실패"
            )
            return ToolResult(
                output={"source": source_name, "candidate_id": candidate_id, "params": params},
                summary=f"실패: fetch({source_name}, {candidate_id}) — {type(e).__name__}: {e}",
                success=False,
                error=f"{type(e).__name__}: {e}",
            )

        # evidence None 처리 (못 찾음)
        if evidence is None:
            return ToolResult(
                output={"source": source_name, "candidate_id": candidate_id,
                        "params": params, "evidence": None},
                summary=f"fetch({source_name}, {candidate_id}): 데이터 없음 (None 반환)",
                success=False,
                error="DataSource가 None 반환 — 해당 조건에 맞는 데이터 없음.",
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
