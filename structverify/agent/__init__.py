"""
structverify.agent — Agentic 검증 시스템.

기존 (v6.14까지):
  - builder_agent: 카탈로그 구축 agent
  - runtime_agent: 검증 파이프라인 오케스트레이션 (Step 3~9)

Phase A 추가:
  - workspace: Agent 작업 공간 (파일 시스템 추상화)
  - memory:    멀티턴 메모리 (markdown append)
  - schemas:   Plan / PlanStep / Observation / AgentVerdict 데이터 모델

Phase B 예정: tools/ (catalog_search, fetch_evidence, calculate, finish)
Phase C 예정: planner.py (Plan Agent)
Phase D 예정: reflector.py + loop.py (멀티턴 실행)
Phase E 예정: verifier 확장 — 여러 data point 받아 계산
Phase F 예정: runtime_agent.py 통합 (Step 7-8을 agent_loop로 교체)
"""

# 기존 export는 그대로 (수정 X)
# 새 모듈은 import하지 않음 (사용자가 명시적으로 from structverify.agent.workspace import ...)
