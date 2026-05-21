"""
agent/runtime_agent.py — Runtime Verification Agent (Agent A)

실시간 검증 요청을 처리하는 메인 Agent. ReAct 패턴 기반.
Thought → Action(Tool Call) → Observation 순환을 통해 파이프라인 제어.

[김예슬 - 2026-04-22]
- Step 3~9 오케스트레이션 전체 담당
- ReAct 패턴으로 각 스텝을 Action으로 정의하고 순차 실행

[김예슬 - 2026-04-23]
- classify_domain() 반환값 튜플 대응: str → (domain, domain_desc)
- config["detected_domain"] → self.config["detected_domain"] 버그 수정
- domain_desc를 self.config에 저장하여 schema_inductor 힌트로 활용

[김예슬 - 2026-04-24]
- induce_schemas Action 설명 업데이트:
  · 기존: HCX-003 generate_json() → JSON 파싱 (실패 가능)
  · 변경: HCX-007 Structured Outputs → JSON Schema 보장 (파싱 실패 없음)
- Action별 사용 모델/API 업데이트:
  · classify_domain   → HCX-DASH-002 (v3 API)
  · score_candidate   → HCX-DASH-002 (v3 API)
  · check_worthiness  → HCX-003 (v1 API)
  · induce_schemas    → HCX-007 Structured Outputs (v3 API)
  · generate_explain  → HCX-003 (v1 API)

[ReAct 패턴 설명]
  LLM이 단순히 답변을 생성하는 것이 아니라, 매 스텝마다 다음을 반복합니다:
    Thought  : "현재 상태에서 무엇을 해야 하는가?" (LLM 내부 추론)
    Action   : 구체적인 도구(함수) 호출
    Observation: 도구 호출 결과를 관찰하고 다음 Thought 수행

  Action → 사용 모델/API 매핑:
    classify_domain   → HCX-DASH-002 (v3, 경량)
    score_candidate   → HCX-DASH-002 (v3, 경량)
    check_worthiness  → HCX-003 (v1, 중량)
    induce_schemas    → HCX-007 Structured Outputs (v3, JSON 보장)
    build_graph       → 내부 로직 (LLM 미사용)
    retrieve_evidence → KOSIS Open API (LLM 미사용)
    verify_claim      → Deterministic 수치 비교 (LLM 미사용)
    generate_explain  → HCX-003 (v1, 중량)

[참고] ReAct (Yao et al., ICLR 2023) — https://github.com/ysymyth/ReAct
"""
from __future__ import annotations

from structverify.core.schemas import (
    Claim, SIRDocument, VerificationResult, GraphNode, GraphEdge, Evidence,
    GraphNodeType, GraphEdgeType,
)
from structverify.detection.domain_classifier import classify_domain
from structverify.detection.claim_detector import detect_claims
from structverify.detection.schema_inductor import induce_schemas
from structverify.graph.graph_builder import build_claim_graph
from structverify.graph.graph_multihop import apply_multihop_verification
from structverify.graph.document_graph import build_document_temporal_graph
from structverify.graph.claim_graph import ClaimGraph
from structverify.retrieval.query_builder import build_query
from structverify.retrieval.evidence_subgraph import build_evidence_subgraph
from structverify.retrieval.kosis_connector import KOSISConnector
from structverify.verification.verifier import verify_claim
from structverify.explanation.explainer import generate_explanation
from structverify.memory import DocumentWorkingMemory  # [머지 이수민 main]
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


class RuntimeAgent:
    """
    Agent A: 실시간 검증 처리.
    Step 3~9를 순차 실행하며 ReAct 패턴으로 파이프라인을 오케스트레이션한다.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        # [v3 김예슬] kosis config에 llm 포함 → CatalogSearchTool LLM Agent가 llm 설정 사용
        kosis_cfg = {
            **self.config.get("kosis", {}),
            "llm": self.config.get("llm", {}),
        }
        self.kosis = KOSISConnector(config=kosis_cfg)

        # [v6.19] Graph Store 초기화 (Neo4j — 옵셔널)
        #   default.yaml graph.store.enabled=true 일 때만 실제 연결.
        #   미설정/미설치/연결실패 시 GraphStore 내부에서 안전하게 비활성화됨.
        from structverify.graph.graph_store import GraphStore
        graph_store_cfg = (self.config.get("graph") or {}).get("store") or {}
        self.graph_store = GraphStore(config=graph_store_cfg)

    async def process(self, sir_doc: SIRDocument) -> tuple[
        list[Claim], list[VerificationResult], list[GraphNode], list[GraphEdge]
    ]:
        """
        SIR 문서 → 전체 검증 파이프라인 실행 (Step 3~9).

        Returns:
            (claims, results, graph_nodes, graph_edges)
        """

        # ── [머지: main의 DocumentWorkingMemory] ──────────────────────
        # 이 doc 처리 동안만 살아있는 in-memory 컨텍스트.
        # v2의 verified_facts(검증값 캐시)와 역할이 다름:
        #   - memory     : doc 단위 — 도메인 가드 / stat_id 캐시 / claim 인덱스
        #   - verified_facts : claim 간 검증값 재사용 (agent loop 내부)
        # 둘은 충돌하지 않으며 상호 보완적이다.
        from uuid import uuid4
        memory = DocumentWorkingMemory(
            doc_id=str(sir_doc.doc_id),
            run_id=str(uuid4())[:8],
            source_uri=getattr(sir_doc, "source_uri", None),
        )

        # ── Action: classify_domain ──────────────────────────────────
        # Tool: HCX-DASH-002 (v3 API, 경량)
        # Thought: "이 문서의 도메인이 무엇인가?"
        # Observation: domain 문자열 + 설명
        domain, domain_desc = await classify_domain(sir_doc, self.config)
        self.config["detected_domain"] = domain
        self.config["detected_domain_desc"] = domain_desc
        memory.record_domain(domain, domain_desc)  # [머지 이수민 main]
        logger.info(f"[Agent A] Step 3 classify_domain → {domain} ({domain_desc})")

        # ── Action: detect_claims ────────────────────────────────────
        # [4-1] candidate_scorer: HCX-DASH-002 (v3, 경량) → 0~1 점수
        # [4-2] claim_detector:   HCX-003 (v1, 중량) → check-worthiness
        # Thought: "검증 가능한 주장 문장을 찾아야 한다"
        # Observation: Claim 객체 리스트
        # TODO [김예슬]: domain-packs 기반 도메인별 few-shot 예시 주입
        claims = await detect_claims(sir_doc, self.config)
        logger.info(f"[Agent A] Step 4 detect_claims → {len(claims)}건")

        if not claims:
            logger.info("[Agent A] 검증 가능한 주장 없음 — 파이프라인 종료")
            return [], [], [], []

        # ── [v4 김예슬] Context Window 부착 ────────────────────────────
        # 각 claim에 앞뒤 문장 context를 붙여서 LLM이 맥락을 이해할 수 있게 함
        # 예: "이는 20년 새 2.6배 증가한 것이다" → 앞 문장 "쉬었음 청년이 21만7천명"도 함께 전달
        # schema_inductor + query_builder에서 context_text 활용
        for claim in claims:
            claim.context_text = _get_context_window(claim, sir_doc, window=2)

        # ── Action: build_document_temporal_graph ─────────────────────
        # Tool: HCX 문서 시간 분석 agent (LLM 1회 호출)
        # Thought: "기사 작성일/anchor_year를 추출해서 '내년/올해' 같은
        #           상대 시점을 절대 연도로 풀 수 있게 해야 한다"
        # Observation: DocumentNode(anchor_year=...) + TemporalExpr 노드
        # [v6.16] 이 단계가 빠져 있어서 anchor_year가 항상 None이던 버그 수정
        temporal_graph = None
        try:
            t_nodes, t_edges = await build_document_temporal_graph(
                sir_doc, self.config
            )
            if t_nodes:
                temporal_graph = ClaimGraph(t_nodes, t_edges)
                # [머지 이수민 main] anchor_year + 시간 표현을 memory에 기록
                for _n in t_nodes:
                    _nt = getattr(getattr(_n, "node_type", None), "value", "")
                    if _nt == "document":
                        _ay = (_n.properties or {}).get("anchor_year")
                        if _ay:
                            try:
                                memory.record_anchor_year(int(_ay))
                            except (ValueError, TypeError):
                                pass
                    elif _nt == "temporal_expr":
                        _expr = (_n.properties or {}).get("expression")
                        _resolved = (_n.properties or {}).get("resolved_value")
                        if _expr and _resolved:
                            memory.record_temporal(str(_expr), str(_resolved))
                logger.info(
                    f"[Agent A] Step 4.5 temporal graph → "
                    f"anchor_year={temporal_graph.get_anchor_year()}"
                )
        except Exception as e:
            logger.warning(f"[Agent A] temporal graph 빌드 실패 (계속 진행): {e}")

        # ── Action: induce_schemas ───────────────────────────────────
        # Tool: HCX-007 Structured Outputs (v3 API)
        # Thought: "각 주장을 indicator/value/unit/population으로 구조화해야 한다"
        # Observation: claim.schema = ClaimSchema({indicator, value, ...})
        # [v4] context_text 포함 → "이는" 같은 대명사 참조 해소
        # [v6.16] temporal_graph 전달 → 상대 시점 해소 + anchor_year fallback
        claims = await induce_schemas(claims, self.config, graph=temporal_graph)
        memory.record_claims(claims)  # [머지 이수민 main] metric_to_claims 인덱싱
        logger.info(
            f"[Agent A] Step 5 induce_schemas → schemas attached "
            f"(memory: {len(memory.metric_to_claims)} metrics)"
        )

        # ── Action: build_claim_graph ────────────────────────────────
        # Tool: 내부 로직 (LLM 미사용)
        # Thought: "ClaimSchema → Knowledge Graph 노드/엣지를 구성해야 한다"
        # Observation: GraphNode[], GraphEdge[]
        # TODO [신준수]: graph_builder.py 노드/엣지 타입 완성
        all_nodes, all_edges = build_claim_graph(claims, sir_doc=sir_doc) # 호출부 로직 변경 [pipeline v3] 김예슬
        logger.info(f"[Agent A] Step 6 build_claim_graph → {len(all_nodes)} nodes")

        # ── Step 7~8: 각 주장별 Evidence 조회 + 검증 ────────────────
        # [Multi-hop v1] Step 9(설명)는 multi-hop 재검증 후로 분리
        #   이유: 파생 주장 검증은 다른 claim들의 Step 8 결과가 모두 필요함
        # [Phase D] config.agent.enabled=true 면 planner+loop 경로 사용:
        #   - planner가 claim마다 Plan 수립 (ReAct: Thought)
        #   - agent_loop가 Plan대로 catalog_search → fetch_evidence → verify 순회
        #   기존 경로(고정 retrieve→verify)는 enabled=false 시 그대로 사용 → 안전 롤백
        # [머지 박재윤 main] asyncio.gather + Semaphore(3) 병렬화.
        #   claim끼리 독립적이므로 agent 경로(planner+loop)도 병렬 안전.
        #   claim 1건당 14~40초 → 8건 직렬이면 ~7분. 병렬 3이면 ~1/3.
        agent_enabled = bool(
            (self.config.get("agent") or {}).get("enabled", False)
        )

        # agent 경로에서 쓸 문서 원문 (planner가 source_text로 사용)
        source_text = self._get_source_text(sir_doc)
        anchor_year = (
            temporal_graph.get_anchor_year() if temporal_graph else None
        )

        import asyncio
        sem = asyncio.Semaphore(3)
        # 병렬 claim들이 memory에 동시 기록 → race 방지용 Lock.
        # DocumentWorkingMemory의 record_* 는 dict 갱신이라 짧지만,
        # record_stat_id_used 같은 복합 갱신을 원자적으로 보호한다.
        mem_lock = asyncio.Lock()

        async def process_one_claim(claim):
            """claim 1건 Step 7~8. 그래프 노드/엣지는 반환만 하고
            병렬 종료 후 메인이 모은다 (extend가 thread-safe하지 않으므로)."""
            claim_nid = f"claim:{claim.claim_id.hex[:8]}"
            async with sem:
                if agent_enabled:
                    # ── [Phase D] Agent Loop 경로 ──────────────────────
                    result, ev_nodes, ev_edges = await self._verify_with_agent(
                        claim, source_text, anchor_year, temporal_graph,
                        claim_nid=claim_nid, memory=memory, mem_lock=mem_lock,
                    )
                    logger.info(
                        f"[Agent A] Step 7~8 agent_loop → {result.verdict.value} "
                        f"(evidence nodes={len(ev_nodes)})"
                    )
                    return result, ev_nodes, ev_edges

                # ── 기존 경로 (고정 retrieve_evidence + verify_claim) ──
                query = build_query(claim)
                evidence, ev_nodes, ev_edges = await build_evidence_subgraph(
                    self.kosis, query, claim_nid,
                )
                logger.info(
                    f"[Agent A] Step 7 retrieve_evidence → "
                    f"{str(evidence)[:80] if evidence else None}"
                )
                # Action: verify_claim (Deterministic, LLM 미개입)
                # 주의: sv2 verify_claim은 memory 파라미터를 받지 않음.
                # 도메인 가드는 호출 후 evidence를 보고 별도로 적용한다.
                result = verify_claim(
                    claim, evidence, self.config, graph=temporal_graph,
                )
                # [머지 이수민 main] 도메인 가드 — evidence가 doc 도메인과
                # 어긋나면 UNVERIFIABLE 강등. evidence.raw_response 또는
                # stat 메타에서 category_path를 찾는다(없으면 가드 통과).
                _ev_cat = None
                if evidence is not None:
                    _ev_cat = (evidence.raw_response or {}).get("category_path")
                if (_ev_cat and not memory.domain_matches_category(_ev_cat)):
                    from structverify.core.schemas import VerdictType
                    logger.warning(
                        f"[Agent A] 도메인 가드 거절: category={_ev_cat!r} "
                        f"vs domain={memory.domain!r} → UNVERIFIABLE"
                    )
                    async with mem_lock:
                        memory.record_stat_id_rejected(
                            str(getattr(evidence, "stat_table_id", "") or "?"),
                            f"도메인 불일치: {memory.domain}",
                        )
                    result.verdict = VerdictType.UNVERIFIABLE
                    result.confidence = min(result.confidence or 0.3, 0.3)
                # [머지 이수민 main] 성공 stat_id를 memory에 캐시
                if (result.verdict.value == "match"
                        and evidence and getattr(evidence, "stat_table_id", None)
                        and claim.schema and claim.schema.indicator):
                    async with mem_lock:
                        memory.record_stat_id_used(
                            indicator=claim.schema.indicator,
                            stat_id=evidence.stat_table_id,
                            category_path=_ev_cat,
                            time_period=getattr(evidence, "time_period", None),
                        )
                logger.info(f"[Agent A] Step 8 verify_claim → {result.verdict.value}")
                return result, ev_nodes, ev_edges

        # ── [Dependency Planning 2026-05-21] level 기반 실행 ──
        # 한 문장에서 분기된 base/derived sub-claim, 또는 같은 indicator를
        # 공유하는 claim들을 *순차 레벨*로 묶어 evidence 재활용.
        #   Level 1 (병렬): base claims
        #   Level 2 (병렬): derived_rate / derived_difference claims
        # Level 간 verified_facts / successful_stat_ids 캐시가 살아 있어 derived가
        # base의 fetch 결과를 자동 재활용. 같은 level 안에선 기존대로 Semaphore(3)
        # 병렬 유지.
        from structverify.agent.dependency_planner import build_execution_levels
        _exec_levels = build_execution_levels(claims)
        logger.info(
            f"[Agent A] dependency planning: {len(_exec_levels)} levels, "
            f"sizes={[len(lvl) for lvl in _exec_levels]}"
        )

        # claim_id → 결과 매핑 (원래 claim 순서대로 정렬 위해)
        _results_by_id: dict[Any, tuple] = {}
        for _lvl_idx, _level_claims in enumerate(_exec_levels):
            if not _level_claims:
                continue
            logger.info(
                f"[Agent A] Level {_lvl_idx + 1}/{len(_exec_levels)}: "
                f"{len(_level_claims)}개 claim 병렬 시작"
            )
            _parallel = await asyncio.gather(
                *[process_one_claim(c) for c in _level_claims]
            )
            for _c, _out in zip(_level_claims, _parallel):
                _results_by_id[_c.claim_id] = _out
            logger.info(
                f"[Agent A] Level {_lvl_idx + 1} 완료 — verified_facts/"
                f"successful_stat_ids 캐시가 다음 level로 전파됨"
            )

        # 원래 claim 순서대로 정렬 (results 인덱스 보존)
        results: list[VerificationResult] = []
        for _c in claims:
            _result, _ev_nodes, _ev_edges = _results_by_id[_c.claim_id]
            results.append(_result)
            all_nodes.extend(_ev_nodes)
            all_edges.extend(_ev_edges)

        # [머지 이수민 main] working memory 통계 로깅
        logger.info(f"[Agent A] working_memory stats: {memory.stats()}")
        if memory.rejected_stat_ids:
            logger.info(
                f"[Agent A] 도메인 가드 거절 stat_id: "
                f"{len(memory.rejected_stat_ids)}건"
            )

        # ── Step 8.5: Multi-hop GraphRAG 파생 주장 재검증 ──────────────
        # Tool: graph_multihop (LLM 미사용 — 비율/배수 계산은 deterministic)
        # Thought: "KOSIS로 직접 검증 못한 파생 주장(2.6배 등)을
        #           COMPARE 엣지 이웃들의 검증된 수치로 재계산할 수 있다"
        # Observation: UNVERIFIABLE 파생 주장 → MATCH/MISMATCH 가능
        results = apply_multihop_verification(
            claims, results, all_edges, self.config
        )

        # ── Step 9: 각 주장별 설명 생성 ────────────────────────────────
        for claim, result in zip(claims, results):
            result.explanation = await generate_explanation(claim, result, self.config)
            logger.info(f"[Agent A] Step 9 generate_explanation → {len(result.explanation or '')}자")

        # ── [v6.19] Step 9.5: Graph Store 영속화 (Neo4j — 옵셔널) ──────
        # graph.store.enabled=true 면 노드/엣지를 Neo4j에 MERGE.
        # 비활성/실패해도 save_graph 내부에서 흡수 — 검증 결과는 그대로 반환.
        if self.graph_store.is_active():
            try:
                n_saved, e_saved = await self.graph_store.save_graph(
                    all_nodes, all_edges
                )
                logger.info(
                    f"[Agent A] Step 9.5 graph_store → "
                    f"Neo4j 저장 노드 {n_saved} / 엣지 {e_saved}"
                )
            except Exception as e:
                logger.warning(f"[Agent A] graph_store 저장 실패 (무시): {e}")

        logger.info(f"[Agent A] 완료: claims={len(claims)}, results={len(results)}, "
                    f"nodes={len(all_nodes)}, edges={len(all_edges)}")

        # [v6.19] job 완료 후 agent_workspace 임시 디렉토리 정리.
        # workspace.cleanup()은 정의돼 있었지만 어디서도 호출되지 않아
        # job마다 agent_workspace/job_* 디렉토리가 무한 누적 →
        # "No space left on device"로 agent_loop 전체가 폴백되는 장애 발생.
        # agent.workspace.persist_after_job=true면 디버깅 위해 보존.
        agent_cfg = self.config.get("agent") or {}
        ws_cfg = dict(agent_cfg.get("workspace") or {})
        if not ws_cfg.get("persist_after_job", False):
            try:
                from structverify.agent.workspace import build_workspace
                # _verify_with_agent와 동일하게 claim.doc_id를 job_id로 사용
                _job_id = ""
                if claims:
                    _job_id = str(getattr(claims[0], "doc_id", "") or "")
                _ws = build_workspace(job_id=_job_id or "job", config=ws_cfg)
                _ws.cleanup()
            except Exception as e:
                logger.warning(f"[Agent A] workspace 정리 실패 (무시): {e}")

        return claims, results, all_nodes, all_edges

    # ── [Phase D] Agent Loop 경로 헬퍼 ──────────────────────────────────────

    def _get_source_text(self, sir_doc: "SIRDocument") -> str:
        """SIR 문서에서 원문 텍스트 복원 — planner의 source_text로 사용."""
        parts: list[str] = []
        for block in getattr(sir_doc, "blocks", []) or []:
            for sent in getattr(block, "sentences", []) or []:
                t = getattr(sent, "text", None)
                if t:
                    parts.append(t)
        return " ".join(parts)

    async def _verify_with_agent(
        self,
        claim: "Claim",
        source_text: str,
        anchor_year: "int | None",
        temporal_graph: "ClaimGraph | None",
        claim_nid: str,
        memory: "DocumentWorkingMemory | None" = None,
        mem_lock=None,
    ) -> "tuple[VerificationResult, list[GraphNode], list[GraphEdge]]":
        """
        [Phase D] planner + agent_loop 으로 claim 1건 검증.

        ReAct:
          Thought  → planner.plan() 이 Plan(검증 전략 + 단계) 수립
          Action   → agent_loop 이 catalog_search → fetch_evidence 순회
          Observation → 각 step 결과 누적
          → AgentVerdict 산출 → VerificationResult 변환

        Returns:
          (VerificationResult, evidence GraphNode 리스트, GraphEdge 리스트)
          — [v6.19] evidence를 그래프에 박아 multihop 재검증이 agent 경로에서도
            동작하게 함. 검증 실패해도 노드는 빈 리스트로 반환(에러 아님).

        실패 시 기존 경로(retrieve_evidence + verify_claim)로 폴백.
        """
        from structverify.agent.planner import Planner, PlannerConfig
        from structverify.agent.loop import agent_loop, LoopConfig
        from structverify.agent.reflect import ReflectAgent, ReflectConfig
        from structverify.agent.workspace import build_workspace
        from structverify.retrieval.registry import build_all_enabled
        import structverify.retrieval.kosis_source  # noqa: F401 — @register_datasource 트리거

        agent_cfg = self.config.get("agent") or {}
        llm_cfg   = self.config.get("llm") or {}

        try:
            # 1) workspace 준비
            ws_cfg = dict(agent_cfg.get("workspace") or {})
            workspace = build_workspace(
                job_id=str(getattr(claim, "doc_id", "") or "job"),
                config=ws_cfg,
            )
            if not workspace.is_initialized():
                workspace.initialize(source_text=source_text or "")
            workspace.create_claim_dir(
                claim.claim_id, claim_data=claim.model_dump(mode="json")
            )

            # 2) DataSource 등록 (KOSIS)
            ds_cfg = self.config.get("data_sources") or {}
            kosis_ds_cfg = dict(ds_cfg.get("kosis") or self.config.get("kosis") or {})
            datasources = {
                ds.name: ds
                for ds in build_all_enabled({
                    "enabled": ["kosis"],
                    "kosis": kosis_ds_cfg,
                })
            }

            # 3) Planner LLM wiring — 기존 LLMClient 재사용
            from structverify.utils.llm_client import LLMClient
            plan_llm = LLMClient(config=llm_cfg)

            async def llm_call_for_plan(prompt: str) -> str:
                return await plan_llm.generate(
                    prompt=prompt,
                    system_prompt="검증 계획 수립 전문가. JSON으로만 답하세요.",
                )

            planner = Planner(
                llm_call=llm_call_for_plan,
                config=PlannerConfig(
                    model=(llm_cfg.get("plan_model") or "HCX-007"),
                    temperature=0.1,
                ),
            )

            # 4) Plan 생성 (ReAct Thought)
            plan = await planner.plan(
                claim, source_text=source_text, anchor_year=anchor_year
            )
            workspace.write_plan(claim.claim_id, plan.model_dump(mode="json"))
            logger.info(
                f"[planner] {claim.claim_id}: Plan 수립 "
                f"type={getattr(plan, 'claim_type', None)} "
                f"steps={len(getattr(plan, 'initial_steps', []) or [])}"
            )

            # 5) Agent Loop 실행 (ReAct Action/Observation)
            loop_cfg = agent_cfg.get("loop") or {}
            loop_mode = str(loop_cfg.get("mode", "deterministic")).strip()
            max_iter = int(loop_cfg.get("max_iterations", 10))

            # [reflect 활성화] mode='reflect'면 ReflectAgent를 loop에 주입.
            #   매 iter LLM이 last_observation을 보고 다음 action을 동적
            #   결정(catalog 결과 부적합 시 검색어 바꿔 재검색, 원문 재독 등).
            #   ReflectAgent는 파싱 실패 시 None을 반환하고, loop은 그 경우
            #   plan의 다음 step으로 deterministic fallback → 안전.
            #   mode='deterministic'이면 reflect_fn=None (기존 동작 유지).
            reflect_fn = None
            if loop_mode == "reflect":
                async def llm_call_for_reflect(prompt: str) -> str:
                    return await plan_llm.generate(
                        prompt=prompt,
                        system_prompt=(
                            "당신은 사실검증 ReAct 에이전트입니다. "
                            "지금까지의 관찰 결과를 보고 다음 action을 "
                            "JSON으로만 답하세요."
                        ),
                    )

                reflect_fn = ReflectAgent(
                    llm_call=llm_call_for_reflect,
                    claim=claim,
                    config=ReflectConfig(),
                    max_iterations=max_iter,
                )
                logger.info(
                    f"[planner] {claim.claim_id}: reflect 모드 활성화 "
                    f"(max_iter={max_iter}) — 매 iter LLM 재계획"
                )

            verdict = await agent_loop(
                plan=plan,
                claim=claim,
                workspace=workspace,
                datasources=datasources,
                config=self.config,
                reflect_fn=reflect_fn,
                loop_config=LoopConfig(
                    max_iterations=max_iter,
                    mode=loop_mode,
                ),
            )

            # 6) AgentVerdict → VerificationResult 변환
            #    [v6.17] agent_loop이 검증에 쓴 KOSIS 데이터(data_points)를
            #    Evidence로 복원 → UI '공식 통계 출처' 박스에 표시됨.
            agent_evidence = None
            _ev_category = None  # 도메인 가드용 — Evidence 스키마엔 없는 필드
            dps = getattr(verdict, "data_points", None) or []
            if dps:
                dp = dps[0]  # 단일 fetch — 첫 data point가 검증 근거
                src = (dp.source or "")
                stat_id = src.split(":", 1)[1] if ":" in src else (src or None)
                _ev_category = getattr(dp, "category_path", None)
                agent_evidence = Evidence(
                    source_name="KOSIS",
                    stat_table_id=stat_id,
                    official_value=dp.resolved_value,
                    unit=dp.resolved_unit,
                    time_period=dp.source_time,
                )

            # [머지 이수민 main] 도메인 가드 — agent 경로에도 적용.
            # data_point의 category_path가 doc 도메인과 어긋나면(예: 인구
            # 기사인데 환경 통계표) verdict를 UNVERIFIABLE로 강등.
            # agent_loop의 표 관련성 체크와 별개의 doc-레벨 안전망.
            if (memory is not None and agent_evidence is not None
                    and _ev_category):
                if not memory.domain_matches_category(_ev_category):
                    from structverify.core.schemas import VerdictType
                    logger.warning(
                        f"[Agent A] 도메인 가드 거절: claim={claim.claim_id} "
                        f"category={_ev_category!r} "
                        f"vs domain={memory.domain!r} → UNVERIFIABLE 강등"
                    )
                    if mem_lock is not None:
                        async with mem_lock:
                            memory.record_stat_id_rejected(
                                str(agent_evidence.stat_table_id or "?"),
                                f"도메인 불일치: {memory.domain}",
                            )
                    verdict.verdict = VerdictType.UNVERIFIABLE
                    verdict.confidence = min(
                        getattr(verdict, "confidence", 0.3) or 0.3, 0.3
                    )
                    agent_evidence = None  # 그래프에도 박지 않음

            # [머지 이수민 main] MATCH면 성공 stat_id를 memory에 캐시
            if (memory is not None and agent_evidence is not None
                    and getattr(verdict.verdict, "value", "") == "match"
                    and getattr(agent_evidence, "stat_table_id", None)
                    and claim.schema and claim.schema.indicator):
                if mem_lock is not None:
                    async with mem_lock:
                        memory.record_stat_id_used(
                            indicator=claim.schema.indicator,
                            stat_id=agent_evidence.stat_table_id,
                            category_path=_ev_category,
                            time_period=getattr(
                                agent_evidence, "time_period", None
                            ),
                        )

            # [v6.19] Evidence → 그래프 노드/엣지 (multihop 재검증 입력)
            ev_nodes, ev_edges = _evidence_to_graph(agent_evidence, claim_nid)

            result = VerificationResult(
                claim_id=claim.claim_id,
                verdict=verdict.verdict,
                confidence=verdict.confidence,
                explanation=verdict.explanation,
                evidence=agent_evidence,
            )
            return result, ev_nodes, ev_edges

        except Exception as e:
            # Agent 경로 실패 → 기존 경로로 안전 폴백
            logger.warning(
                f"[Agent A] agent_loop 실패 → 기존 경로 폴백: {e}"
            )
            query = build_query(claim)
            evidence, fb_nodes, fb_edges = await build_evidence_subgraph(
                self.kosis, query, claim_nid,
            )
            fb_result = verify_claim(
                claim, evidence, self.config, graph=temporal_graph
            )
            return fb_result, fb_nodes, fb_edges


# ── [v6.19] Evidence → 그래프 노드/엣지 변환 ────────────────────────────────

def _evidence_to_graph(
    evidence: "Evidence | None", claim_nid: str,
) -> "tuple[list[GraphNode], list[GraphEdge]]":
    """agent 경로가 얻은 Evidence를 그래프 노드/엣지로 변환.

    기존 경로의 build_evidence_subgraph와 동일한 모양:
      EVIDENCE 노드 1개 + (claim ─VERIFIED_BY→ evidence) 엣지 1개
    이렇게 박아야 Step 8.5 multihop이 COMPARE 이웃의 검증 수치를
    그래프에서 찾아 파생 주장을 재검증할 수 있다.

    evidence가 None이거나 official_value가 없으면 빈 리스트
    (검증 근거가 없으므로 그래프에 박을 것도 없음).
    """
    if evidence is None or getattr(evidence, "official_value", None) is None:
        return [], []

    stat_id = getattr(evidence, "stat_table_id", None) or "unknown"
    ev_node_id = f"evidence:{claim_nid}:{stat_id}"
    ev_node = GraphNode(
        node_id=ev_node_id,
        node_type=GraphNodeType.EVIDENCE,
        label=f"{evidence.source_name or 'KOSIS'}({stat_id})",
        properties={
            "official_value": evidence.official_value,
            "unit": getattr(evidence, "unit", None),
            "time_period": getattr(evidence, "time_period", None),
            "stat_table_id": stat_id,
            "source_name": evidence.source_name,
        },
    )
    ev_edge = GraphEdge(
        from_node=claim_nid,
        to_node=ev_node_id,
        edge_type=GraphEdgeType.VERIFIED_BY,
    )
    return [ev_node], [ev_edge]


# ── [v4 김예슬] Context Window 헬퍼 ─────────────────────────────────────────

def _get_context_window(
    claim: "Claim",
    sir_doc: "SIRDocument",
    window: int = 2,
) -> str:
    """
    claim의 앞 문장 window개를 SIR Tree에서 가져와서 context 문자열로 반환.

    [v4 김예슬 - 2026-05-07]
    "이는 20년 새 2.6배 증가한 것이다" 같은 문장은
    앞 문장 "2024년 쉬었음 청년이 21만7천명이다"가 있어야 정확한 schema 추출 가능.

    SIR Tree의 block_id/sent_id를 기반으로 같은 블록 내 앞 문장들을 찾음.
    NEXT_SENT 엣지를 graph에서 탐색하지 않고 sir_doc에서 직접 조회 (더 효율적).

    Args:
        claim: 대상 주장
        sir_doc: SIR 문서 (blocks → sentences 계층)
        window: 앞에서 가져올 문장 수 (기본 2)

    Returns:
        "앞문장1. 앞문장2. 현재문장" 형태의 context 문자열
    """
    target_block = claim.block_id
    target_sent  = claim.sent_id

    # sir_doc에서 해당 블록 찾기
    block = None
    for b in sir_doc.blocks:
        if b.block_id == target_block:
            block = b
            break

    if not block or not block.sentences:
        return claim.claim_text

    # 현재 문장 인덱스 찾기
    sent_idx = None
    for i, sent in enumerate(block.sentences):
        if sent.sent_id == target_sent:
            sent_idx = i
            break

    if sent_idx is None:
        return claim.claim_text

    # 앞 window개 문장 수집
    start = max(0, sent_idx - window)
    context_sents = []
    for i in range(start, sent_idx + 1):
        text = block.sentences[i].text.strip()
        if text:
            context_sents.append(text)

    return " ".join(context_sents)