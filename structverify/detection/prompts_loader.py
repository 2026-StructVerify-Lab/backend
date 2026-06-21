"""detection/prompts_loader.py — domain-pack YAML 로드·few-shot 헬퍼.

domain_classifier.py의 _load_domain_pack()에서 분리 (로직 move-only).

[김예슬 - 2026-04-22] domain-packs/{domain}/prompts.yaml
TODO [김예슬]: claim_detector/candidate_scorer few-shot 주입은 #13 이후 연결
"""
from __future__ import annotations

import os
from typing import Any

import yaml

from structverify.utils.logger import get_logger

logger = get_logger(__name__)


def domain_packs_dir(config: dict | None) -> str:
    """config에서 domain-packs 루트 경로."""
    cfg = config or {}
    return cfg.get("domain_packs_dir", "domain-packs")


def prompts_yaml_path(domain: str, config: dict | None) -> str:
    """domain-packs/{domain}/prompts.yaml 절대/상대 경로."""
    return os.path.join(domain_packs_dir(config), domain, "prompts.yaml")


def load_domain_pack(domain: str, config: dict | None = None) -> dict[str, Any] | None:
    """
    domain-packs/{domain}/prompts.yaml 로드.
    없으면 None 반환 (에러 아님).
    """
    yaml_path = prompts_yaml_path(domain, config)

    if not os.path.exists(yaml_path):
        logger.debug(f"Domain Pack 없음: {yaml_path}")
        return None

    try:
        with open(yaml_path, encoding="utf-8") as f:
            pack = yaml.safe_load(f)
        logger.info(f"Domain Pack 로드: {yaml_path}")
        if isinstance(pack, dict):
            return pack
        return None
    except Exception as e:
        logger.warning(f"Domain Pack 로드 실패: {yaml_path} — {e}")
        return None


def load_domain_prompts(domain: str, config: dict | None = None) -> dict[str, Any] | None:
    """load_domain_pack 별칭 — claim_detector TODO 명칭과 동일."""
    return load_domain_pack(domain, config)


def few_shot_examples_from_pack(
    pack: dict[str, Any] | None,
    *,
    section: str = "few_shot_examples",
) -> list[Any]:
    """
    pack dict에서 few-shot 예시 리스트 추출.

    section 키가 없으면 빈 리스트 (동작 변경 없이 주입 전 단계용).
    """
    if not pack:
        return []
    raw = pack.get(section)
    if isinstance(raw, list):
        return raw
    return []


def format_few_shot_block(examples: list[Any]) -> str:
    """
    few-shot 예시를 프롬프트에 붙일 블록 문자열로 변환.

    examples 원소: str 또는 {"input": ..., "output": ...} dict.
    """
    if not examples:
        return ""
    lines: list[str] = ["", "[도메인 few-shot 예시]"]
    for i, ex in enumerate(examples, 1):
        if isinstance(ex, str):
            lines.append(f"  {i}. {ex}")
        elif isinstance(ex, dict):
            inp = ex.get("input") or ex.get("sentence") or ""
            out = ex.get("output") or ex.get("label") or ""
            lines.append(f"  {i}. input={inp!r} → {out!r}")
        else:
            lines.append(f"  {i}. {ex!r}")
    return "\n".join(lines)


def inject_few_shot(
    base_prompt: str,
    pack: dict[str, Any] | None,
    *,
    section: str = "few_shot_examples",
) -> str:
    """base_prompt 뒤에 pack의 few-shot 블록을 붙인다. pack 없으면 원문 그대로."""
    block = format_few_shot_block(few_shot_examples_from_pack(pack, section=section))
    if not block:
        return base_prompt
    return base_prompt.rstrip() + block + "\n"


def resolve_prompt_for_step(
    base_prompt: str,
    domain: str | None,
    config: dict | None,
    *,
    step: str,
) -> str:
    """
    domain-pack에서 step별 few-shot을 찾아 base_prompt에 붙인다.

    키 우선순위: {step}_few_shot → {step}_examples → {step} → few_shot_examples
    pack/예시 없으면 base_prompt 그대로 (기존 동작 유지).
    """
    if not domain:
        return base_prompt
    pack = load_domain_pack(domain, config)
    if not pack:
        return base_prompt
    for section in (f"{step}_few_shot", f"{step}_examples", step, "few_shot_examples"):
        if few_shot_examples_from_pack(pack, section=section):
            return inject_few_shot(base_prompt, pack, section=section)
    return base_prompt
