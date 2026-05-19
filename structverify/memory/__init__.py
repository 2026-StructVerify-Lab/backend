# [이수민 - 2026-05-13]
# - AgentMemory 모듈 패키지 진입점 (초안)
# - 그래프 누적 KG 방향 — Phase 2로 보류
#
# [이수민 - 2026-05-14 — 방향 재정렬]
# - 작업 목표가 "정확도 개선용 사례 메모리(exemplar retrieval)"로 재조정됨
#   목적:
#     · 도메인 분류 정확도 ↑ (Step 3)
#     · 시간 해석 정확도 ↑ ("작년", "올해" 등의 해석, Step 4.5)
# - 신규 노출: ExemplarStore, DomainExample, TemporalExample, get_embedding
# - 기존 AgentMemory 골격(agent_memory.py 등)은 보존하지만 import 안 함
#   (Phase 2에서 그래프 누적 KG 합칠 때 다시 활용 예정)
"""
structverify.memory — 정확도 개선용 사례 메모리 + (보류) 그래프 누적 KG

[현재 활성]  exemplar_store.py / embedder.py
[Phase 2]   agent_memory.py / normalizer.py / storage/ (그래프 누적)
"""
from structverify.memory.working_memory import (
    DocumentWorkingMemory,
    StatIdUsage,
)
from structverify.memory.exemplar_store import (
    ExemplarStore,
    DomainExample,
    TemporalExample,
    format_domain_hint,
    format_temporal_hint,
)
from structverify.memory.embedder import get_embedding, EMBEDDING_DIM

__all__ = [
    # ★ 현재 활성: Document-scoped Working Memory
    "DocumentWorkingMemory",
    "StatIdUsage",
    # Phase 2 보류: 영속 사례 retrieval
    "ExemplarStore",
    "DomainExample",
    "TemporalExample",
    "format_domain_hint",
    "format_temporal_hint",
    "get_embedding",
    "EMBEDDING_DIM",
]
