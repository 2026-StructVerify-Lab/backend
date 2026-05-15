"""
agent/executor.py — 스텝 실행 래퍼 (Executor)

에이전틱 루프에서 각 스텝을 실행하는 통일된 진입점.
기존 레이어 함수를 그대로 호출하고, RunContext의 hint를 주입한다.

- 담당자: 신준수
"""
# 수정자: 신준수
# 수정 날짜: 2026-05-15
# 수정 내용: 에이전틱 리팩토링 - Executor 레이어 래핑 모듈 신규

# [DONE] execute_step() 단일 진입점 구현
# [DONE] Step 5 / 7 / 8 / 9 분기
# [DONE] ctx.hints[step] → 각 레이어 함수에 hint 전달
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structverify.detection.schema_inductor import induce_schema_for_claim
from structverify.explanation.explainer import generate_explanation
from structverify.retrieval.evidence_subgraph import build_evidence_subgraph
from structverify.retrieval.query_builder import build_query
from structverify.verification.verifier import verify_claim
from structverify.utils.logger import get_logger

if TYPE_CHECKING:
    from structverify.agent.context import RunContext
    from structverify.graph.claim_graph import ClaimGraph
    from structverify.memory.working_memory import DocumentWorkingMemory
    from structverify.retrieval.kosis_connector import KOSISConnector

logger = get_logger(__name__)


async def execute_step(
    step: int,
    ctx: "RunContext",
    *,
    kosis: "KOSISConnector",
    full_graph: "ClaimGraph",
    memory: "DocumentWorkingMemory",
    config: dict,
) -> Any:
    """
    스텝 번호에 따라 해당 레이어 함수를 호출하고 결과를 반환한다.

    ctx.hints[step]이 있으면 각 레이어 함수에 hint로 전달한다.
    Step 7에서 생성된 ev_nodes/ev_edges는 ctx.local_nodes/local_edges에 누적한다.
    (공유 리스트에 즉시 append하지 않음 — T1 해결)

    Args:
        step:       실행할 스텝 번호 (5 / 7 / 8 / 9)
        ctx:        현재 claim의 RunContext
        kosis:      KOSISConnector 인스턴스
        full_graph: ClaimGraph (verifier/explainer용)
        memory:     DocumentWorkingMemory (도메인 가드용)
        config:     파이프라인 설정 dict

    Returns:
        스텝별 반환값:
          5 → ClaimSchema | None
          7 → Evidence | None
          8 → VerificationResult
          9 → str (explanation)
    """
    hint = ctx.hints.get(step)

    # ── Step 5: schema 재유도 ──────────────────────────────────────────────
    if step == 5:
        # 목적: 롤백 후 단일 claim에 대해 schema를 1개 재추출.
        #       hint가 있으면 indicator/source_phrase 타겟 명시로 LLM 유도.
        schema = await induce_schema_for_claim(
            ctx.claim,
            config=config,
            graph=full_graph,
            hint=hint,
        )
        if schema is not None:
            ctx.claim.schema = schema
        logger.info(
            f"[Executor] Step 5 claim={ctx.claim.sent_id} "
            f"indicator={schema.indicator if schema else None}"
        )
        return schema

    # ── Step 7: KOSIS 증거 검색 ───────────────────────────────────────────
    elif step == 7:
        # 목적: 공유 리스트(all_nodes/edges) 대신 ctx.local 버퍼에 누적 (T1 해결).
        #       롤백 시 clear_local_buffers()로 초기화하면 오염 없이 재시도 가능.
        claim_nid = f"claim:{ctx.claim.claim_id.hex[:8]}"
        query = build_query(ctx.claim)
        evidence, ev_nodes, ev_edges = await build_evidence_subgraph(
            kosis, query, claim_nid,
        )
        ctx.local_nodes.extend(ev_nodes)
        ctx.local_edges.extend(ev_edges)

        # stat_id 캐시 (DocumentWorkingMemory 연동 — 기존 그대로)
        if (
            evidence and evidence.stat_table_id
            and ctx.claim.schema and ctx.claim.schema.indicator
        ):
            cached = memory.get_stat_id_for_indicator(ctx.claim.schema.indicator)
            if not cached:
                memory.record_stat_id_used(
                    indicator=ctx.claim.schema.indicator,
                    stat_id=evidence.stat_table_id,
                    category_path=getattr(evidence, "category_path", None),
                    time_period=getattr(evidence, "time_period", None),
                )

        logger.info(
            f"[Executor] Step 7 claim={ctx.claim.sent_id} "
            f"evidence={str(evidence)[:80] if evidence else None}"
        )
        return evidence

    # ── Step 8: 팩트 판별 (Deterministic, LLM 미사용) ─────────────────────
    elif step == 8:
        evidence = ctx.snapshots[7].output if 7 in ctx.snapshots else None
        result = verify_claim(
            ctx.claim, evidence, config,
            graph=full_graph, memory=memory,
        )
        logger.info(
            f"[Executor] Step 8 claim={ctx.claim.sent_id} "
            f"verdict={result.verdict.value}"
        )
        return result

    # ── Step 9: 설명 생성 (LLM) ───────────────────────────────────────────
    elif step == 9:
        result = ctx.snapshots[8].output if 8 in ctx.snapshots else None
        if result is None:
            logger.warning(f"[Executor] Step 9 — Step 8 결과 없음, 건너뜀")
            return None
        explanation = await generate_explanation(ctx.claim, result, config)
        result.explanation = explanation
        logger.info(
            f"[Executor] Step 9 claim={ctx.claim.sent_id} "
            f"explanation={len(explanation or '')}자"
        )
        return explanation

    else:
        raise ValueError(f"[Executor] 알 수 없는 스텝: {step}")
