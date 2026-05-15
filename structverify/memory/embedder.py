# [이수민 - 2026-05-14]
# - HCX 임베딩 API 호출 헬퍼 (사례 저장/검색 공용)
# - retrieval/catalog_search.py:_get_embedding 로직 재활용
# - exemplar_store가 직접 HTTP 호출하지 않고 이 모듈을 통하도록 일원화
# - KOSIS catalog와 같은 임베딩 모델 → 같은 벡터 공간에서 검색 일관성 확보
# - 구조화 단계 — 시그니처와 인터페이스만. 본문은 다음 단계에서 구현.
"""
memory/embedder.py — 사례용 텍스트 임베딩 헬퍼

[설계 의도]
- 기존 KOSIS catalog 검색이 쓰는 HCX 임베딩(v2)을 재활용
- 같은 임베딩 모델이어야 catalog와 같은 의미 공간에서 검색 가능
- 벡터 차원: HCX-emb v2 = 1024차원

[참조]
- 원본 구현: structverify/retrieval/catalog_search.py:_get_embedding (line 230~)
- 엔드포인트: https://clovastudio.stream.ntruss.com/v1/api-tools/embedding/v2
"""
from __future__ import annotations

from structverify.utils.logger import get_logger

logger = get_logger(__name__)


EMBEDDING_DIM = 1024  # HCX-emb v2 차원


async def get_embedding(text: str, api_key: str | None = None) -> list[float] | None:
    """
    텍스트 → HCX 임베딩 벡터 (1024-dim).

    Args:
        text:    임베딩 대상 텍스트 (도메인 분류용 본문 일부 또는 시간 표현+문맥)
        api_key: CLOVASTUDIO_API_KEY (None이면 환경변수에서 읽음)

    Returns:
        list[float] (길이 1024) 또는 None (API 실패 시)
    """
    raise NotImplementedError  # 다음 단계


async def get_embeddings_batch(
    texts: list[str], api_key: str | None = None
) -> list[list[float] | None]:
    """
    여러 텍스트 한 번에 임베딩 (배치).

    초기 사례 일괄 적재 시 사용. 단건은 get_embedding() 권장.

    Args:
        texts: 임베딩 대상 텍스트 리스트

    Returns:
        list[list[float] | None] — 입력 순서 보존, 실패는 None
    """
    raise NotImplementedError


# ── 보조 ────────────────────────────────────────────────────────────────────

def vector_to_pgvector_str(vec: list[float]) -> str:
    """
    list[float] → pgvector INSERT용 문자열 "[1.0,2.0,...]".

    pgvector는 vector 타입을 문자열 리터럴로 받음.
    catalog_search.py:_search_pgvector의 변환 로직과 동일.
    """
    raise NotImplementedError
