"""
detection/domain_classifier.py — 도메인 자동 분류 (Step 3)

입력 텍스트의 도메인을 판별하고 적절한 Domain Pack을 선택한다.

[김예슬 - 2026-04-22]
- DOMAIN_CLASSIFY_PROMPT: few-shot 예시 3개 포함한 프롬프트로 교체
- classify_domain(): model_tier="light"(HCX-DASH-001)로 명시적 지정
- _build_text_preview(): 블록 타입 고려한 미리보기 텍스트 구성
- _load_domain_pack(): domain-packs/{domain}/prompts.yaml 로드 시도

[김예슬 - 2026-04-23 v1]
- SUPPORTED_DOMAINS 하드코딩 제거 → domain-packs/ 디렉토리 기반으로 변경

[김예슬 - 2026-04-23 v1]
- DomainRegistry 클래스 추가: 레지스트리 기반 도메인 관리
  · 문제: LLM 자유 생성 시 같은 주제를 다른 이름으로 분류하는 파편화 발생
    (예: real_estate / housing_market / property → 모두 같은 주제)
  · 해결: 기존 등록 도메인 목록 + 설명을 LLM 프롬프트에 주입
    → LLM이 기존 도메인 중 유사한 게 있으면 재사용, 없으면 신규 생성
  · 신규 도메인 생성 시 레지스트리(registry.yaml)에 자동 저장
- classify_domain() 반환값: str → tuple[str, str] (domain, description)
  · domain: 도메인 키 (영문 소문자)
  · description: 도메인 한국어 설명 (레지스트리에서 조회 또는 신규 생성)
- DOMAIN_CLASSIFY_PROMPT: 기존 도메인 목록 동적 주입 방식으로 변경

[참고] ReAct (Yao et al., ICLR 2023)
  Agent의 첫 단계로 도메인을 판별하여 이후 전략을 결정하는 패턴
"""
from __future__ import annotations

from structverify.core.schemas import SIRDocument
from structverify.detection.domain.classify import _classify_domain_with_llm
from structverify.detection.domain.registry import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_SEED_DOMAINS,
    DOMAIN_NAME_PATTERN,
    DomainRegistry,
)
from structverify.detection.prompts_loader import load_domain_pack
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


# ── 메인 진입점 ──────────────────────────────────────────────────────────────
async def classify_domain(
    sir_doc: SIRDocument,
    config: dict | None = None,
) -> tuple[str, str]:
    """
    SIR 문서의 도메인을 LLM으로 분류한다.

    반환값이 (domain, description) 튜플로 바뀐 이유:
      - domain만 반환하면 나중에 설명을 다시 조회해야 함
      - 한 번에 받아서 schema_inductor 등에서 도메인 힌트로 바로 활용 가능

    분류 로직:
      1) 레지스트리에서 기존 도메인 목록 + 설명 로드
      2) LLM 프롬프트에 목록 주입 → 기존 재사용 or 신규 생성
      3) 신규 도메인이면 레지스트리에 자동 저장
      4) confidence 낮으면 "general" fallback

    Args:
        sir_doc: 분류할 SIR 문서
        config: 설정 dict

    Returns:
        (domain, description) 튜플
        예: ("agriculture", "농림수산식품 (농가, 경작면적, ...)")
    """
    domain, description = await _classify_domain_with_llm(sir_doc, config)
    sir_doc.detected_domain = domain
    load_domain_pack(domain, config)
    return domain, description
