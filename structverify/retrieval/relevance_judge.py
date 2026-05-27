"""
structverify.retrieval.relevance_judge — P32: LLM 기반 표 관련성 판단.

배경:
  kosis_connector._is_table_relevant()는 *토큰 매칭 룰베이스*라 sweet spot 밖
  케이스에 false negative 발생. 예:
    - indicator: "체외 충격파 쇄석술 장비 수"
    - table_name: "시군구(서울인천경기강원)별 주요 의료장비 현황"
    - 룰: "장비"(2글자, len<3 룰에 막힘), "쇄석술"(table에 없음) → 거부 ❌
    - 의미적으로는 *정답 표* (장비 row가 의료장비 분류에 포함될 가능성 높음)

설계:
  룰베이스를 *fast-path*로 유지하되, 거부 결정 시 *LLM에게 한 번 더 위임*.
  LLM은 claim 원문 + schema + table_name을 보고 *의미적 매칭* 판단.
  룰 *통과* 시엔 LLM 호출 없음 (속도).

  - 룰 통과 → 그대로 진행 (LLM 호출 X)
  - 룰 거부 → LLM 가드 호출 → True/False → 최종 판단

config:
  kosis.relevance_guard.{enabled, llm_fallback, model_tier}
"""
from __future__ import annotations

import json
import re
from typing import Any

from structverify.utils.logger import get_logger

logger = get_logger(__name__)


def _build_prompt(
    claim_text: str,
    indicator: str,
    population: str,
    parent_path: str,
    table_name: str,
) -> str:
    return f"""당신은 KOSIS 통계표 관련성 reviewer입니다. 사용자 claim의 검증에
*이 표가 의미적으로 적합한가*를 한 줄로 판단하세요.

[사용자 claim]
- 원문: {claim_text or '(없음)'}
- 검증 지표 (indicator): {indicator or '(없음)'}
- 대상 집단/지역 (population): {population or '(없음)'}
- 카테고리 경로 (parent_path): {parent_path or '(없음)'}

[후보 KOSIS 통계표]
- 표 이름: {table_name}

[판단 기준]
- 표 이름이 indicator를 *포함하거나*, indicator의 *상위 분류* (예: "체외 충격파 쇄석술 장비"의 상위 = "의료장비")로 *직접 매칭*되면 → relevant.
- 표 이름이 population (예: "강원도", "시도별")을 포함/매칭하면 → relevant 가능성 ↑.
- 표 이름이 더 *광범위*해도 (예: "주요 의료장비"가 "체외 충격파 쇄석술 장비"의 상위) row 검색으로 정답 찾을 가능성 있으면 → relevant.
- 외국 국가/해외 통계 / 장래 추계·전망 / 다른 도메인 (인구↔경제↔기상)이면 → NOT relevant.
- parent_path와 표 이름이 같은 분류 트리에 있으면 → relevant.

[응답 형식 — JSON only]
{{
  "relevant": true | false,
  "reason": "한 줄 이유"
}}
"""


def _parse(raw: str) -> tuple[bool | None, str]:
    """LLM 응답 파싱. (relevant, reason). None이면 파싱 실패."""
    try:
        m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if not m:
            return None, ""
        data = json.loads(m.group(0))
        rel = data.get("relevant")
        reason = str(data.get("reason") or "")
        if isinstance(rel, bool):
            return rel, reason
        if isinstance(rel, str):
            return rel.strip().lower() in ("true", "yes", "y", "1"), reason
        return None, reason
    except Exception as e:
        logger.debug(f"[relevance_judge] 파싱 실패: {e}")
        return None, ""


async def is_table_relevant_semantic(
    *,
    claim_text: str,
    indicator: str,
    population: str,
    parent_path: str,
    table_name: str,
    config: dict | None,
) -> tuple[bool | None, str]:
    """LLM 기반 표 관련성 판단.

    Args:
        claim_text: claim 원문 (1~3문장).
        indicator: schema.indicator.
        population: schema.population (지역/집단).
        parent_path: schema.parent_path (계층 카테고리).
        table_name: KOSIS 표 이름.
        config: 전체 config dict.

    Returns:
        (relevant, reason).
        relevant=True: 의미적 매칭 OK, fetch 진행 권장.
        relevant=False: 매칭 안 됨, 거부 권장.
        relevant=None: LLM 호출 실패/파싱 실패 → 호출자가 보수적으로 처리.
    """
    if not table_name or (not indicator and not claim_text):
        return None, "input 부족"

    _cfg = (config or {}).get("kosis") or {}
    _rg = _cfg.get("relevance_guard") or {}
    model_tier = str(_rg.get("model_tier") or "light").strip().lower()

    prompt = _build_prompt(claim_text, indicator, population, parent_path, table_name)
    from structverify.utils.llm_client import LLMClient
    llm = LLMClient(config=(config or {}).get("llm") or {})
    try:
        raw = await llm.generate(
            prompt=prompt,
            system_prompt="KOSIS 표 관련성 reviewer. JSON만 응답.",
            model_tier=model_tier,
        )
    except Exception as e:
        logger.warning(f"[relevance_judge] LLM 호출 실패: {e}")
        return None, f"llm_error: {e}"

    rel, reason = _parse(raw)
    if rel is None:
        logger.info(f"[relevance_judge] 파싱 실패 raw={raw[:200]!r}")
        return None, "parse_failed"

    logger.info(
        f"[relevance_judge] table={table_name!r} vs indicator={indicator!r}: "
        f"relevant={rel} reason={reason[:120]!r}"
    )
    return rel, reason
