"""
structverify.retrieval.base — DataSource 추상 인터페이스.

KOSIS만이 아니라 *어떤 데이터 소스*든 (회사 DB, CSV 파일, 외부 API)
*동일한 인터페이스*로 플러그인 가능하도록 설계.

회사 사용 시나리오:
  - 글로벌 회사: World Bank API + OECD API
  - 사내 데이터: PostgreSQL/Snowflake + S3 CSV
  - 한국 미디어: KOSIS (현재 default)

Config (config/default.yaml):
    data_sources:
      enabled: ["kosis", "custom_csv"]
      kosis: {...}
      custom_csv: {...}

Registry로 등록:
    from structverify.retrieval.registry import register_datasource

    @register_datasource("my_internal_db")
    class MyInternalDB(BaseDataSource):
        async def search_catalog(self, query, ...): ...
        async def fetch_evidence(self, candidate_id, ...): ...
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


# ── Result 타입 (dict 호환, 가볍게) ──────────────────────────────

class CatalogCandidate(dict):
    """카탈로그 검색 결과 1건.

    필수 키:
        id          : 후보 식별자 (예: 'DT_1L9U108')
        name        : 표/지표 이름
        score       : 검색 점수 (0~1)
    선택 키:
        category    : 분류 (예: '인구/가구')
        description : 설명
        ...source-specific 필드
    """


class EvidenceData(dict):
    """fetch 결과 raw 데이터.

    필수 키:
        value           : 수치값 (float)
        unit            : 단위
        time_period     : 시점 (YYYY 또는 YYYY-MM)
    선택 키:
        official_value  : 정규화된 값
        raw_response    : API 원본 응답 (디버깅용)
        rows            : 전체 row 목록 (KOSIS 같은 표 데이터)
        ...
    """


# ── 추상 인터페이스 ───────────────────────────────────────────────

class BaseDataSource(ABC):
    """
    모든 데이터 소스의 공통 인터페이스.

    필수 구현:
        - search_catalog: 키워드로 표/지표 후보 검색
        - fetch_evidence: 후보 1개의 실제 데이터 조회

    선택 구현 (기본은 False / no-op):
        - supports_time_filter: API가 시점 파라미터 지원하는지
        - supports_population_filter: 인구 필터 지원 여부
        - close: 리소스 정리 (DB 커넥션 등)
    """

    name: str = "base"
    """Registry에 등록될 이름. @register_datasource로 자동 설정."""

    @abstractmethod
    async def search_catalog(
        self,
        query: str,
        category: list[str] | None = None,
        top_k: int = 10,
        context: dict[str, Any] | None = None,
    ) -> list[CatalogCandidate]:
        """카탈로그 검색.

        Args:
            query: 검색 키워드 (자연어 또는 키워드 조합).
            category: 분류 힌트 (예: ['인구', '가구']).
            top_k: 반환할 최대 후보 수.
            context: [P31 2026-05-22] source-specific 부가 정보. 검색 정확도 보강용.
                권장 키 (source가 알 만한 것만 사용):
                    - 'parent_path': str — schema의 계층 카테고리 (예: "보건 > 의료자원 > 의료장비")
                    - 'raw_claim':   str — 원문 claim 텍스트 (앞 200자)
                    - 'population':  str — schema의 대상 집단/지역
                    - 'indicator':   str — schema의 지표명
                source가 모르는 키는 무시. KOSIS는 LLM 카테고리 추출 시 활용.

        Returns:
            CatalogCandidate 리스트. score 내림차순 권장.
        """

    @abstractmethod
    async def fetch_evidence(
        self,
        candidate_id: str,
        params: dict[str, Any] | None = None,
    ) -> EvidenceData | None:
        """실제 데이터 조회.

        Args:
            candidate_id: search_catalog 결과의 'id'.
            params: source-specific 파라미터.
                예 (KOSIS): {'prdSe': 'M', 'startPrdDe': '202504'}.
                예 (CSV):  {'column': 'sales', 'row_filter': 'region=KR'}.

        Returns:
            EvidenceData 또는 None (못 찾으면).
        """

    # ── 선택 기능 (기본 false / no-op) ─────────────────────────

    def supports_time_filter(self) -> bool:
        """API가 시점 파라미터 필터링을 지원하는지.

        True면 fetch_evidence에 시점 params 전달 가능.
        False면 모든 row 가져온 후 verifier가 필터링.
        """
        return False

    def supports_population_filter(self) -> bool:
        """인구/카테고리 필터 지원 여부."""
        return False

    async def close(self) -> None:
        """리소스 정리. DB 커넥션 등."""
        return None

    async def get_table_meta(
        self,
        candidate_id: str,
        meta_type: str = "ITM",
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """[P30 2026-05-22] 표의 *항목/분류 메타* 조회 — *데이터는 X*.

        용도: catalog_search → fetch_evidence 사이에서, 표 이름만으론 정답인지
        불확실할 때 LLM이 *표 내부 구조*(어떤 항목/분류 코드)를 보고 판단하기
        위한 가벼운 메타 호출.

        Args:
            candidate_id: search_catalog 결과의 'id' (stat_id 등).
            meta_type: source-specific 메타 타입.
                KOSIS: "ITM" (통계항목), "OBJL01"~"OBJL08" (분류 항목), "PRD", "CMMT".

        Returns:
            메타 dict 또는 list (source 형식 그대로). 실패/미지원이면 None.
        """
        return None


# ── 헬퍼: 다중 source orchestration (Phase B+에서 활용) ───────────

class MultiSourceRouter:
    """
    여러 DataSource를 묶어서 *순차/병렬* 검색.

    Phase B에서 tool wrapper가 이 클래스 사용:
      - config.data_sources.enabled = ["kosis", "custom_db"] 면 *둘 다* 검색
      - 결과 합쳐서 ranked list 반환
    """

    def __init__(self, sources: list[BaseDataSource]):
        self.sources = sources

    async def search_catalog_all(
        self,
        query: str,
        category: list[str] | None = None,
        top_k_per_source: int = 5,
    ) -> dict[str, list[CatalogCandidate]]:
        """각 source 검색 결과 dict로."""
        results: dict[str, list[CatalogCandidate]] = {}
        for src in self.sources:
            try:
                results[src.name] = await src.search_catalog(
                    query, category=category, top_k=top_k_per_source
                )
            except Exception as e:
                # 한 source 실패해도 나머지 진행
                results[src.name] = []
        return results

    async def close_all(self) -> None:
        for src in self.sources:
            try:
                await src.close()
            except Exception:
                pass