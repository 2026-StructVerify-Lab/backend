"""detection/domain/classify.py — 도메인 LLM 분류 실행.

domain_classifier.py에서 분리 (로직 move-only, 동작 변경 없음).
"""
from __future__ import annotations

from structverify.core.schemas import SIRDocument
from structverify.detection.domain.preview import _build_text_preview
from structverify.detection.domain.registry import (
    DEFAULT_SEED_DOMAINS,
    DOMAIN_NAME_PATTERN,
    DomainRegistry,
)
from structverify.detection._config import (
    domain_confidence_threshold,
    domain_registry_path,
    model_tier_for,
)
from structverify.detection._llm import get_llm_client
from structverify.detection.prompts.domain import DOMAIN_CLASSIFY_PROMPT
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


async def _classify_domain_with_llm(
    sir_doc: SIRDocument,
    config: dict | None = None,
) -> tuple[str, str]:
    """레지스트리 + LLM으로 (domain, description) 반환."""
    config = config or {}
    registry_path = domain_registry_path(config)
    registry = DomainRegistry(registry_path)

    preview = _build_text_preview(sir_doc)
    domain_list_str = registry.format_for_prompt()
    llm = get_llm_client(config)

    try:
        result = await llm.generate_json(
            prompt=DOMAIN_CLASSIFY_PROMPT.format(
                domain_list=domain_list_str,
                text_preview=preview,
            ),
            system_prompt="도메인 분류 전문가. JSON으로만 답하세요.",
            model_tier=model_tier_for(config, "domain_classify", default="light"),
        )

        raw_domain    = result.get("domain", "general")
        description   = result.get("description", "")
        is_new        = bool(result.get("is_new", False))
        confidence    = float(result.get("confidence", 0.0))
        reason        = result.get("reason", "")

        # 도메인 형식 검증
        if not DOMAIN_NAME_PATTERN.match(raw_domain):
            logger.warning(f"도메인 형식 오류 '{raw_domain}' → general")
            raw_domain, description = "general", DEFAULT_SEED_DOMAINS["general"]

        # confidence 낮으면 general
        if confidence < domain_confidence_threshold(config):
            logger.warning(f"confidence 낮음 ({confidence:.2f}) → general")
            raw_domain, description = "general", DEFAULT_SEED_DOMAINS["general"]

        # 신규 도메인이면 레지스트리에 저장
        if is_new and raw_domain != "general":
            registry.register(raw_domain, description)

        # 기존 도메인이면 레지스트리의 공식 설명 사용 (LLM 설명이 다를 수 있음)
        if not is_new:
            registered = registry.load()
            description = registered.get(raw_domain, description)

        domain = raw_domain
        logger.info(
            f"도메인 분류: {domain} ({'신규' if is_new else '기존'}) "
            f"confidence={confidence:.2f}, reason={reason}"
        )

    except Exception as e:
        logger.error(f"도메인 분류 실패: {e}")
        domain, description = "general", DEFAULT_SEED_DOMAINS["general"]

    return domain, description
