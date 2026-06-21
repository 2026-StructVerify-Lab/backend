"""detection/domain/registry.py — 도메인 레지스트리 (registry.yaml).

domain_classifier.py에서 분리 (로직 move-only, 동작 변경 없음).

[김예슬 - 2026-04-23] DomainRegistry — LLM 도메인 파편화 방지
"""
from __future__ import annotations

import os
import re

import yaml

from structverify.utils.logger import get_logger

logger = get_logger(__name__)

# [김예슬 - 2026-04-23] DomainRegistry — LLM 도메인 파편화 방지
# confidence_threshold 기본값 → detection/config.yaml (domain.confidence_threshold)
CONFIDENCE_THRESHOLD = 0.6  # re-export 호환; 런타임은 config.domain_confidence_threshold()
DOMAIN_NAME_PATTERN = re.compile(r"^[a-z][a-z_]{0,29}$")

# 기본 시드 도메인 — 레지스트리 파일이 없을 때 초기값으로 사용
DEFAULT_SEED_DOMAINS: dict[str, str] = {
    "agriculture":  "농림수산식품 (농가, 경작면적, 수확량, 축산, 어업)",
    "economy":      "경제/경기 (GDP, 성장률, 소비, 수출입, 물가, 산업생산)",
    "finance":      "금융/증권 (금리, 환율, 주가, 대출, 가계부채, 보험)",
    "population":   "인구/가구 (출생, 사망, 혼인, 고령화, 인구구조)",
    "employment":   "고용/노동/임금 (취업률, 실업률, 임금, 근로시간)",
    "healthcare":   "보건/의료 (질병, 의료기관, 사망률, 건강보험)",
    "education":    "교육 (학생, 학교, 교육비, 진학률, 입시)",
    "policy":       "정책/행정 (예산, 법률, 복지, 지원금, 정부)",
    "environment":  "환경/에너지 (기후, 탄소, 재생에너지, 환경오염)",
    "general":      "분류 불가 또는 복합 도메인",
}


class DomainRegistry:
    """
    도메인 레지스트리 — {domain: description} 매핑을 파일로 영속 관리.

    registry.yaml 구조:
        agriculture: "농림수산식품 (농가, 경작면적, ...)"
        economy: "경제/경기 (GDP, 성장률, ...)"
        real_estate: "부동산 (아파트, 매매가, ...)"   ← 런타임에 추가됨

    사용법:
        registry = DomainRegistry("domain-packs/registry.yaml")
        domains = registry.load()           # {domain: description} 반환
        registry.register("real_estate", "부동산 관련 통계")
    """

    def __init__(self, registry_path: str = "domain-packs/registry.yaml"):
        self.registry_path = registry_path

    def load(self) -> dict[str, str]:
        """
        레지스트리 파일 로드.
        파일이 없으면 DEFAULT_SEED_DOMAINS를 파일로 저장 후 반환.
        """
        if not os.path.exists(self.registry_path):
            logger.info(f"레지스트리 없음 → 시드 도메인으로 초기화: {self.registry_path}")
            self._save(DEFAULT_SEED_DOMAINS)
            return dict(DEFAULT_SEED_DOMAINS)

        try:
            with open(self.registry_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return {k: str(v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"레지스트리 로드 실패 → 시드 사용: {e}")
            return dict(DEFAULT_SEED_DOMAINS)

    def register(self, domain: str, description: str) -> None:
        """
        새 도메인을 레지스트리에 추가하고 파일로 저장.
        이미 있으면 무시.
        """
        current = self.load()
        if domain in current:
            return

        current[domain] = description
        self._save(current)
        logger.info(f"새 도메인 등록: {domain} — {description}")

    def _save(self, data: dict[str, str]) -> None:
        os.makedirs(os.path.dirname(self.registry_path) or ".", exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=True)

    def format_for_prompt(self) -> str:
        """
        프롬프트 주입용 문자열 생성.
        예: "- agriculture: 농림수산식품 (농가, 경작면적, ...)"
        """
        domains = self.load()
        lines = [f"- {k}: {v}" for k, v in sorted(domains.items())]
        return "\n".join(lines)
