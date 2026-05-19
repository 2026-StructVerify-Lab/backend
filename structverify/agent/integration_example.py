"""structverify.agent.integration_example — runtime_agent에 agent loop를 끼우는 *예시 코드*.

★ 이 파일은 *작동하는 코드가 아니라 가이드*. 사용자가 자신의 runtime_agent.py에 *수동 통합*.

## 통합 단계

### 1. config 확장 (`config/default.yaml`)

```yaml
agent:
  enabled: false              # ← true로 켜야 agent loop 사용
  llm:
    plan_model: "HCX-007"
    plan_temperature: 0.1
  loop:
    max_iterations: 10
    mode: "deterministic"     # phase D는 이거만. Phase E에서 "reflect" 추가.

data_sources:
  enabled: ["kosis"]
  default_source: "kosis"
  kosis:
    # 기존 KOSIS 설정 (api_key 등) 여기로 통합
    api_key: null  # env var에서 자동 로드 가정
```

### 2. runtime_agent.py에 *config 분기* 추가

기존 Step 7-8 (retrieve_evidence + verify_claim)을 *agent loop로 대체*:

```python
# runtime_agent.py (사용자 코드)

class RuntimeAgent:
    async def process(self, sir_doc):
        # ... Steps 1-6 그대로 (도메인 분류, 클레임 탐지, 스키마 유도, 그래프 빌드)

        for claim in claims:
            if self.config.agent.enabled:
                # ★ NEW: Agent loop 경로
                result = await self._verify_with_agent(claim, source_text, anchor_year)
            else:
                # 기존 경로 (Step 7-8)
                evidence = await self._retrieve_evidence(claim)
                result = await self._verify_claim(claim, evidence)
                result.explanation = await generate_explanation(claim, result, self.config)

            results.append(result)
        # ...

    async def _verify_with_agent(self, claim, source_text, anchor_year):
        '''Phase D: agent loop으로 검증.'''
        from structverify.agent.planner import Planner, PlannerConfig
        from structverify.agent.loop import agent_loop, LoopConfig
        from structverify.agent.workspace import build_workspace
        from structverify.retrieval.registry import build_all_enabled

        # 1. workspace 준비
        workspace = build_workspace(
            job_id=str(self.job_id),
            config=self.config.agent.workspace.model_dump() if hasattr(self.config.agent.workspace, 'model_dump') else dict(self.config.agent.workspace),
        )
        if not workspace.is_initialized():
            workspace.initialize(source_text=source_text)
        workspace.create_claim_dir(claim.claim_id, claim_dict=claim.model_dump(mode="json"))

        # 2. DataSource 등록 (KOSIS만 우선)
        from structverify.retrieval import kosis_source  # noqa — register_datasource 트리거
        datasources = {
            ds.name: ds for ds in build_all_enabled({
                "enabled": ["kosis"],
                "kosis": dict(self.config.data_sources.kosis) if hasattr(self.config.data_sources, 'kosis') else {},
            })
        }

        # 3. Planner — LLM call wiring
        async def llm_call_for_plan(prompt):
            # ★ 사용자 wiring: HCX client 호출
            # 예시 (실제 HCX 호출 방식에 맞게 수정):
            from structverify.llm.hcx_client import call_hcx  # ← 실제 모듈/함수 이름 확인
            return await call_hcx(
                prompt=prompt,
                model=self.config.agent.llm.plan_model,
                temperature=0.1,
            )

        planner = Planner(
            llm_call=llm_call_for_plan,
            config=PlannerConfig(
                model=self.config.agent.llm.plan_model,
                temperature=0.1,
            ),
        )

        # 4. Plan 생성
        plan = await planner.plan(claim, source_text=source_text, anchor_year=anchor_year)
        workspace.write_plan(claim.claim_id, plan.model_dump(mode="json"))

        # 5. Loop 실행
        verdict = await agent_loop(
            plan=plan,
            claim=claim,
            workspace=workspace,
            datasources=datasources,
            config=self.config.model_dump() if hasattr(self.config, 'model_dump') else dict(self.config),
            reflect_fn=None,  # Phase D = deterministic. Phase E에서 추가.
            loop_config=LoopConfig(
                max_iterations=self.config.agent.loop.max_iterations,
                mode="deterministic",
            ),
        )

        # 6. AgentVerdict → 기존 VerificationResult 변환
        # (사용자 코드의 result schema에 맞게 변환)
        result = self._agent_verdict_to_result(claim, verdict)
        return result

    def _agent_verdict_to_result(self, claim, agent_verdict):
        '''AgentVerdict → 기존 VerificationResult 변환.

        agent_verdict.verdict ('match' | 'mismatch' | ...) → result.verdict
        agent_verdict.explanation → result.explanation
        agent_verdict.confidence → result.confidence

        evidence는 마지막 fetch_evidence observation에서 복원 가능 (workspace.read 등).
        '''
        # ★ 사용자 코드의 VerificationResult schema에 맞게 작성
        from structverify.core.schemas import VerificationResult, VerdictType
        return VerificationResult(
            claim_id=claim.claim_id,
            verdict=VerdictType(agent_verdict.verdict.value),
            confidence=agent_verdict.confidence,
            explanation=agent_verdict.explanation,
            evidence=None,  # 또는 마지막 observation에서 복원
        )
```

### 3. 첫 테스트 (안전 모드)

`config.yaml`에서:
```yaml
agent:
  enabled: true  # 켜기
```

같은 출생아 기사 재실행 후 로그 확인:
- `[planner] {claim_id}: Plan 생성 완료. type=... data_points=...`
- `[loop] {claim_id}: 시작. plan.type=growth_rate, steps=4, mode=deterministic`
- `[loop] {claim_id} iter 1: action=catalog_search`
- `[loop] {claim_id} iter 2: action=fetch_evidence`
- ... 등

만약 *Plan은 잘 생성되는데 Loop이 fail*하면:
- KOSISDataSource의 TODO 4곳 (`structverify/retrieval/kosis_source.py`) 사용자 코드와 매핑 확인
- 특히 catalog_search/kosis_connector의 *실제 함수 이름 + 시그니처*

## 디버깅 팁

1. **Plan 단계 실패** → planner 직접 호출해서 LLM 응답 확인:
   ```python
   plan = await planner.plan(claim, ...)
   print(plan.model_dump_json(indent=2))
   ```

2. **DataSource 호출 실패** → KOSIS adapter 직접 테스트:
   ```python
   from structverify.retrieval.kosis_source import KOSISDataSource
   ds = KOSISDataSource()
   cands = await ds.search_catalog(query="출생아 수")
   print(cands)
   ```

3. **agent.enabled=false로 즉시 롤백** — 기존 경로로 돌아감.

## 점진적 도입 추천

1. **Step 1**: agent.enabled=false 유지. Phase A-C zip만 적용 — *기존 결과 그대로*.
2. **Step 2**: *별도 테스트 스크립트*로 planner.plan() 호출 → Plan JSON 확인.
3. **Step 3**: kosis_source.py의 TODO 4곳 채움 + KOSIS DataSource 단독 테스트.
4. **Step 4**: agent.enabled=true로 한 claim만 처리 → 로그 분석.
5. **Step 5**: 8 claim 전체 실행. 기존 결과와 비교.
"""
