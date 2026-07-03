"""structverify.retrieval.custom_csv_source — Custom CSV DataSource.

회사 업로드 CSV를 BaseDataSource 인터페이스로 노출.
가정 CSV 형태: indicator,year,region,value,unit (행 = 시점별 값).

레퍼런스: retrieval/kosis_source.py (KOSISDataSource).
search_catalog: 키워드 매칭 (임베딩/DB 없음). fetch_evidence: 다음 단계.
"""
from __future__ import annotations

import csv
import os
import re
from typing import Any

from .base import BaseDataSource, CatalogCandidate, EvidenceData
from .registry import register_datasource
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

# CSV 컬럼 → 표준 필드 기본 매핑 (config.column_mapping 으로 덮어쓰기)
_DEFAULT_COLUMN_MAPPING = {
    "indicator": "indicator",
    "time_period": "year",
    "region": "region",
    "value": "value",
    "unit": "unit",
}


@register_datasource("custom_csv")
class CustomCSVDataSource(BaseDataSource):
    """회사 업로드 CSV 데이터소스.

    config 예 (config/default.yaml의 data_sources.custom_csv):
        base_path: "./uploads/custom"
        catalog_json: "tables.csv"          # CSV 파일명 (base_path 기준)
        column_mapping: {value: "val", ...} # CSV 컬럼명 커스텀 (선택)
        csv_path: "/abs/path.csv"           # (테스트 편의) CSV 경로 직접 주입
    """

    name = "custom_csv"

    def __init__(self, **config: Any):
        self.config = config
        self.base_path = config.get("base_path", "")
        self.catalog_json = config.get("catalog_json", "")
        # 테스트 편의: csv_path 직접 주입 우선. 없으면 base_path/catalog_json로 해석.
        self.csv_path = config.get("csv_path") or self._resolve_csv_path()
        # CSV 컬럼명 매핑 — config 우선, 없으면 기본
        self.column_mapping = {
            **_DEFAULT_COLUMN_MAPPING,
            **(config.get("column_mapping") or {}),
        }
        logger.info(
            f"[CustomCSVDataSource] 초기화: csv_path={self.csv_path!r}, "
            f"column_mapping={self.column_mapping}"
        )

    # ── CSV 헬퍼 ──

    def _resolve_csv_path(self) -> str:
        """base_path/catalog_json 으로 CSV 경로 해석."""
        if self.base_path and self.catalog_json:
            return os.path.join(self.base_path, self.catalog_json)
        return self.catalog_json or self.base_path or ""

    def _read_rows(self) -> list[dict[str, str]]:
        """CSV 전체 행을 dict 리스트로. 경로 없거나 못 읽으면 []."""
        if not self.csv_path or not os.path.exists(self.csv_path):
            logger.warning(f"[CustomCSVDataSource] CSV 경로 없음: {self.csv_path!r}")
            return []
        try:
            with open(self.csv_path, encoding="utf-8") as f:
                return list(csv.DictReader(f))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[CustomCSVDataSource] CSV 읽기 실패: {e}")
            return []

    @staticmethod
    def _match_score(query: str, indicator: str) -> float:
        """query↔indicator 단순 매칭 점수.

        완전일치 1.0 / 부분포함 0.8 / 토큰 겹침 비례(0~0.5) / 없으면 0.0.
        (is_table_relevant 만큼 정교할 필요 없음 — 단순 substring/토큰.)
        """
        q, ind = query.strip().lower(), indicator.strip().lower()
        if not q or not ind:
            return 0.0
        if q == ind:
            return 1.0
        if q in ind or ind in q:
            return 0.8
        qtok, itok = set(re.findall(r"\w+", q)), set(re.findall(r"\w+", ind))
        common = qtok & itok
        return 0.5 * len(common) / len(qtok | itok) if common else 0.0

    # ── BaseDataSource 인터페이스 ──

    async def search_catalog(
        self,
        query: str,
        category: list[str] | None = None,
        top_k: int = 10,
        context: dict[str, Any] | None = None,
    ) -> list[CatalogCandidate]:
        """CSV indicator 컬럼을 query로 키워드 매칭 → 후보 반환.

        같은 (indicator, region)은 시점별 여러 행이 있어도 후보 1개로 묶음.
        반환: [{"id": "<indicator>|<region>", "name": "<indicator> (<region>)",
                "score": <매칭점수>}] — score 내림차순, 매칭 없으면 [].
        """
        ind_col = self.column_mapping["indicator"]
        reg_col = self.column_mapping["region"]

        # (indicator, region) 단위로 묶어 표 1개 = 후보 1개
        groups: dict[tuple[str, str], float] = {}
        for row in self._read_rows():
            indicator = (row.get(ind_col) or "").strip()
            if not indicator:
                continue
            region = (row.get(reg_col) or "").strip()
            score = self._match_score(query, indicator)
            if score <= 0.0:
                continue
            groups.setdefault((indicator, region), score)  # 같은 그룹 첫 점수 유지

        candidates: list[CatalogCandidate] = [
            {
                "id": f"{ind}|{reg}",
                "name": f"{ind} ({reg})" if reg else ind,
                "score": sc,
            }
            for (ind, reg), sc in groups.items()
        ]
        candidates.sort(key=lambda c: c["score"], reverse=True)
        logger.info(
            f"[CustomCSVDataSource] search_catalog(query={query!r}): "
            f"{len(candidates)}개 후보"
        )
        return candidates[:top_k]

    @staticmethod
    def _year_key(year: Any) -> int:
        """연도 정렬용 키. 파싱 실패 시 -1 (가장 뒤)."""
        try:
            return int(str(year).strip())
        except (TypeError, ValueError):
            return -1

    async def fetch_evidence(
        self,
        candidate_id: str,
        params: dict[str, Any] | None = None,
        workspace: Any = None,
        **kwargs: Any,
    ) -> EvidenceData | None:
        """candidate_id("지표|지역") + params["time_period"]로 CSV 행 매칭 → EvidenceData.

        time_period 매칭: year 컬럼과 문자열 정규화 비교 ("2024" == 2024).
        time_period 없으면 가장 최근 연도 행. 맞는 행 없으면 None.
        """
        params = params or {}
        indicator, _, region = candidate_id.partition("|")  # "지표|지역" 분해
        indicator, region = indicator.strip(), region.strip()

        ind_col = self.column_mapping["indicator"]
        reg_col = self.column_mapping["region"]
        year_col = self.column_mapping["time_period"]
        val_col = self.column_mapping["value"]
        unit_col = self.column_mapping["unit"]

        # 해당 indicator(+region) 행만 추림
        rows = [
            r for r in self._read_rows()
            if (r.get(ind_col) or "").strip() == indicator
            and (r.get(reg_col) or "").strip() == region
        ]
        if not rows:
            return None

        tp = params.get("time_period")
        tp_str = str(tp).strip() if tp is not None else ""
        if tp_str:
            matched = next(
                (r for r in rows if str(r.get(year_col, "")).strip() == tp_str),
                None,
            )
        else:
            matched = max(rows, key=lambda r: self._year_key(r.get(year_col)))

        if matched is None:
            return None

        try:
            value = float((matched.get(val_col) or "").strip())
        except (TypeError, ValueError):
            logger.warning(
                f"[CustomCSVDataSource] value 파싱 실패: {matched.get(val_col)!r}"
            )
            return None

        return {
            "value": value,
            "unit": (matched.get(unit_col) or "").strip(),
            "time_period": str(matched.get(year_col, "")).strip(),
            "source": "custom_csv",
            "indicator": indicator,
            "region": region,
            "matched_row": matched,
        }
