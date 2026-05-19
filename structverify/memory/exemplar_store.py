# [이수민 - 2026-05-14]
# - 정확도 개선용 사례 메모리 (exemplar retrieval store)
# - 목적:
#   · 도메인 분류 정확도 ↑ — 과거 (텍스트, 정답 도메인) 사례 retrieve → few-shot hint
#   · 시간 해석 정확도 ↑ — 과거 (표현, 문맥, anchor_year, 정답 resolved) 사례 retrieve
# - 저장 단위: DomainExample, TemporalExample (둘 다 임베딩 포함)
# - 검색 방식: pgvector cosine 유사도 (기존 KOSIS catalog 검색과 동일 패턴)
# - 임베딩: catalog_search.py의 HCX 임베딩 API 재활용
# - 구조화 단계 — 시그니처와 인터페이스만. 본문은 다음 단계에서 구현.
"""
memory/exemplar_store.py — 사례 기반 retrieval memory

[설계 의도]
LLM이 매번 추론(도메인 분류, 시간 풀이)을 처음부터 하지 않고,
과거 검증된 사례를 retrieve해서 few-shot hint로 주입.

[라이프사이클]
  실행 후: pipeline 끝에 정답 케이스 → store_domain() / store_temporal()
  실행 중: Step 3 (domain), Step 4.5 (temporal)에서 retrieve_* 호출 → 프롬프트 hint

[저장 위치]
  PostgreSQL의 별도 테이블 (pgvector extension 이미 활성화됨):
    - domain_examples (text_embedding, domain, ...)
    - temporal_examples (expression_embedding, context, anchor_year, resolved, ...)

[Phase 2 — 통합 예정]
  classify_domain():
    exemplars = await store.retrieve_domain_examples(text, top_k=3)
    prompt += "참고 사례: ..."
    → LLM 호출

  build_document_temporal_graph():
    for expr in temporal_expressions:
        exemplars = await store.retrieve_temporal_examples(expr, anchor_year, top_k=3)
        prompt_hint += "..."
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from structverify.utils.logger import get_logger

logger = get_logger(__name__)


# ── 사례 데이터 모델 ────────────────────────────────────────────────────────

class DomainExample(BaseModel):
    """
    도메인 분류 사례.

    저장 시점: pipeline 끝, 검증 결과(verdict)가 정답으로 확인된 케이스만.
    """
    example_id:  str
    text:        str                    # 원문 일부 (제목 + 처음 N문장)
    embedding:   list[float] | None = None
    domain:      str                    # 정답 도메인 (employment, environment 등)
    confidence:  float = 1.0            # 사람 검수 / LLM 자체평가
    created_at:  datetime = Field(default_factory=datetime.utcnow)
    source_doc:  str | None = None      # 출처 doc_id (추적용)


class TemporalExample(BaseModel):
    """
    시간 표현 해석 사례.

    저장 시점: document_graph LLM agent 결과가 검증된 케이스.
    """
    example_id:  str
    expression:  str                    # 원본 표현 ("작년", "재작년 같은 기간")
    context:     str                    # 표현이 등장한 문장 + 앞뒤 1문장
    embedding:   list[float] | None = None  # expression + context 임베딩
    anchor_year: int                    # 그 문서의 anchor_year
    resolved:    str                    # 정답 절대 시점 ("2024", "2022-01..2022-11")
    basis:       str | None = None      # 풀이 근거
    created_at:  datetime = Field(default_factory=datetime.utcnow)
    source_doc:  str | None = None


# ── Exemplar Store API ──────────────────────────────────────────────────────

class ExemplarStore:
    """
    pgvector 기반 사례 저장/검색.

    [라이프사이클]
        store = ExemplarStore()
        await store.connect()

        # 저장 (pipeline 후)
        await store.store_domain(text, domain, ...)
        await store.store_temporal(expression, context, anchor, resolved, ...)

        # 검색 (pipeline 중)
        exs = await store.retrieve_domain_examples(text, top_k=3)
        exs = await store.retrieve_temporal_examples(expr, anchor, top_k=3)
    """

    def __init__(self, config: dict | None = None):
        """
        Args:
            config: {
                "pg_dsn": "postgresql://...",
                "embedding_dim": 1024,   # HCX 임베딩 차원
            }
        """
        self.config = config or {}

    async def connect(self) -> None:
        """asyncpg 풀 생성 + 테이블 존재 확인."""
        raise NotImplementedError  # 다음 단계

    async def close(self) -> None:
        """연결 종료."""
        raise NotImplementedError  # 다음 단계

    # ── 저장 ────────────────────────────────────────────────────────────────

    async def store_domain(
        self,
        text: str,
        domain: str,
        confidence: float = 1.0,
        source_doc: str | None = None,
    ) -> str:
        """
        도메인 사례 저장.

        흐름:
          ① text → 임베딩 (HCX API)
          ② DomainExample 객체 생성
          ③ domain_examples 테이블에 INSERT

        Returns:
            example_id
        """
        raise NotImplementedError

    async def store_temporal(
        self,
        expression: str,
        context: str,
        anchor_year: int,
        resolved: str,
        basis: str | None = None,
        source_doc: str | None = None,
    ) -> str:
        """
        시간 표현 사례 저장.

        흐름:
          ① (expression + context) → 임베딩
          ② TemporalExample 객체 생성
          ③ temporal_examples 테이블에 INSERT

        Returns:
            example_id
        """
        raise NotImplementedError

    # ── 검색 ────────────────────────────────────────────────────────────────

    async def retrieve_domain_examples(
        self,
        text: str,
        top_k: int = 3,
    ) -> list[DomainExample]:
        """
        주어진 텍스트와 의미적으로 가까운 도메인 사례 retrieve.

        흐름:
          ① text → 임베딩
          ② pgvector cosine 유사도 검색
          ③ top_k 사례 반환 (domain 분포 다양성 확보 위해 필요 시 re-rank)

        Returns:
            list[DomainExample] — 유사도 높은 순
        """
        raise NotImplementedError

    async def retrieve_temporal_examples(
        self,
        expression: str,
        context: str | None = None,
        anchor_year: int | None = None,
        top_k: int = 3,
    ) -> list[TemporalExample]:
        """
        주어진 시간 표현과 비슷한 과거 사례 retrieve.

        흐름:
          ① (expression + context or expression only) → 임베딩
          ② pgvector 검색 (anchor_year 가까운 케이스 우대)
          ③ top_k 사례 반환

        Args:
            expression:  새 문서에서 추출된 시간 표현
            context:     주변 문맥 (있으면 정확도 ↑)
            anchor_year: 새 문서의 anchor_year (있으면 비슷한 anchor 사례 우대)

        Returns:
            list[TemporalExample]
        """
        raise NotImplementedError

    # ── 유틸 ────────────────────────────────────────────────────────────────

    async def count(self) -> dict[str, int]:
        """저장된 사례 수 통계. {domain: N, temporal: M}"""
        raise NotImplementedError


# ── 프롬프트 hint 변환 헬퍼 ─────────────────────────────────────────────────
# retrieve 결과를 LLM 프롬프트에 주입할 텍스트로 변환.
# 도메인 분류 / 시간 해석에서 공통으로 쓰임.

def format_domain_hint(examples: list[DomainExample]) -> str:
    """
    DomainExample 리스트 → 프롬프트에 끼울 few-shot 텍스트.

    예시 출력:
        # 참고 사례 (과거 분류 결과)
        - "쉬었음 청년이 21만..." → employment (신뢰도 0.95)
        - "연평균기온 14.8도..."   → environment (신뢰도 0.92)
    """
    raise NotImplementedError


def format_temporal_hint(examples: list[TemporalExample]) -> str:
    """
    TemporalExample 리스트 → 프롬프트 hint 텍스트.

    예시 출력:
        # 참고 사례 (과거 시간 표현 풀이)
        - "작년" (anchor=2024 문서) → 2023
        - "재작년 같은 기간" (anchor=2024) → 2022-01..2022-11
    """
    raise NotImplementedError
