"""
structverify.agent.tools.query_rewriter — P30: catalog query 일반화.

배경:
  catalog_search는 표 *이름* 임베딩만 보므로, query가 *row-level keyword* (예:
  "체외 충격파 쇄석술 장비 수")면 표 이름과 매칭 단어가 거의 없어 정답 표가
  catalog 후보에 진입조차 못 한다. *표 이름 친화 어휘*로 query를 변형하면
  같은 의미를 *표 제목 키워드* (예: "시군구별 의료장비 현황")로 표현 가능.

용도:
  catalog_search Tool에서 query_rewrite=true 옵션 시 호출. 원본 query + claim
  컨텍스트 → 변형 query 후보 N개. 각 변형을 catalog_search에 돌려서 합집합을
  반환 → 정답 표 진입률 ↑.

호출자가 LLM 변형 시점/횟수를 제어할 수 있도록 *순수 함수*로 제공.
"""
from __future__ import annotations

import json
import re
from typing import Any

from structverify.utils.logger import get_logger

logger = get_logger(__name__)


def _build_prompt(query: str, claim_info: dict, n: int) -> str:
    return f"""당신은 KOSIS 통계표 검색어 일반화 도우미입니다. 사용자 query가
*세부 row-level 키워드* (예: "체외 충격파 쇄석술기")인 경우, KOSIS 표 *이름*에
주로 쓰이는 *상위 분류 어휘*로 변형하세요. KOSIS 표 이름 패턴 예시:

  - "시군구별 ○○ 현황", "시도별 ○○ 보유 현황"
  - "주요 ○○ 통계", "○○ 보유율"
  - "기관 종별 ○○", "지역별 ○○"

[원본 query]
{query!r}

[claim 컨텍스트]
- indicator: {claim_info.get('indicator')!r}
- population (지역/대상): {claim_info.get('population')!r}
- time_period: {claim_info.get('time_period')!r}
- unit: {claim_info.get('unit')!r}

[변형 규칙]
1. 원본 query의 *세부 항목*은 *상위 카테고리*로 일반화 (예: "체외 충격파 쇄석술 장비" → "의료장비")
2. population이 지역이면 "시군구별", "시도별" 같은 *분류 어휘* 추가
3. 표 제목에 흔한 단어 ("현황", "보유", "주요") 한두 개 자연스럽게 포함
4. 의미는 보존하되 *catalog 표 이름과 매칭 확률이 높은 어휘* 사용
5. 서로 *겹치지 않는 다양한 angle*의 {n}개 변형 생성 (단순 단어 순서 바꾸기 X)

[응답 형식 — JSON only, 다른 텍스트 금지]
{{
  "variations": ["변형1", "변형2", ...]  // 정확히 {n}개
}}
"""


def _parse_variations(raw: str, n: int) -> list[str]:
    """LLM 응답에서 variations list 추출. 실패 시 빈 list."""
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return []
        data = json.loads(m.group(0))
        vs = data.get("variations") or []
        if not isinstance(vs, list):
            return []
        # 정규화 + 중복 제거
        out: list[str] = []
        seen: set[str] = set()
        for v in vs:
            if not isinstance(v, str):
                continue
            s = v.strip().strip("\"'")
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out[:n]
    except Exception as e:
        logger.debug(f"[query_rewriter] 파싱 실패: {e}")
        return []


async def rewrite_query(
    *,
    query: str,
    claim: Any,
    config: dict | None,
) -> list[str]:
    """원본 query + claim → 변형 query 후보 list.

    Args:
        query: 원본 catalog 검색어.
        claim: Claim 객체 (schema 추출용).
        config: 전체 config dict. config.catalog_search.query_rewriter.{n_variations, model_tier} 사용.

    Returns:
        변형 query list (LLM 실패 시 []). 호출자가 각 변형으로 catalog_search 재호출.
    """
    if not query or not query.strip():
        return []

    _cfg = (config or {}).get("catalog_search") or {}
    _qr = _cfg.get("query_rewriter") or {}
    n = int(_qr.get("n_variations") or 3)
    model_tier = str(_qr.get("model_tier") or "light").strip().lower()

    _schema = getattr(claim, "schema", None) if claim is not None else None
    claim_info = {
        "indicator": (getattr(_schema, "indicator", None) or "") if _schema else "",
        "population": (getattr(_schema, "population", None) or "") if _schema else "",
        "time_period": (getattr(_schema, "time_period", None) or "") if _schema else "",
        "unit": (getattr(_schema, "unit", None) or "") if _schema else "",
    }

    prompt = _build_prompt(query, claim_info, n)
    from structverify.utils.llm_client import LLMClient
    llm = LLMClient(config=(config or {}).get("llm") or {})
    try:
        raw = await llm.generate(
            prompt=prompt,
            system_prompt="KOSIS 검색어 일반화 도우미. JSON만 응답.",
            model_tier=model_tier,
        )
    except Exception as e:
        logger.warning(f"[query_rewriter] LLM 호출 실패: {e}")
        return []

    variations = _parse_variations(raw, n)
    if variations:
        logger.info(f"[query_rewriter] {query!r} → {variations}")
    return variations
