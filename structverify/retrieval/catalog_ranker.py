"""
structverify.retrieval.catalog_ranker — LLM batch ranking.

배경:
  기존 indicator semantic guard(키워드 룰) + relevance_judge(per-table T/F) 두 가드를
  *후보 N개 한 번에 비교하는 LLM ranking*으로 통합. 이유:
    - 키워드 룰은 한정자가 사전에 없는 도메인(투석/혈관조영 등) 확장 불가.
    - per-table 판단은 *비교*가 필요한 한정자 매칭(특수의료장비 vs 의료장비)을 잡지 못함.
    - N표를 한 번에 비교하면 한정자 매칭 + 부분집합/상위집합 관계 판별이 자연스러움.

설계:
  - input: claim(indicator/population/parent_path/원문) + candidates[{id, name, org, ...}]
  - output: [{id, score, reason}, ...] (score 0~1, 높은 순 정렬)
  - 호출자가 score 임계치(예: 0.15) 미만은 reject, 나머지는 score 순서로 try.
  - LLM 1회 호출 (메타데이터는 표 이름 + org_name + category_path까지만 노출 — 과적합 회피).

config:
  data_sources.kosis.catalog_ranker.{enabled, score_threshold, model_tier}
"""
from __future__ import annotations

import json
import re
from typing import Any

from structverify.utils.logger import get_logger

logger = get_logger(__name__)

# LLM 응답이 표 5~25개 ranking이라 ~1KB 정도. v3 light tier 충분.
_DEFAULT_MODEL_TIER = "light"
_DEFAULT_SCORE_THRESHOLD = 0.15  # 이 미만은 reject


def _normalize_candidate_for_prompt(c: dict) -> dict:
    """LLM 프롬프트용으로 후보 메타데이터를 안전 추출. raw StatRecord 핸들링.

    Returns:
        {id, name, org_name, category_path}. 비어있는 필드는 ''.
    """
    cid = str(c.get("id") or "")
    name = str(c.get("name") or "")
    # name에 "[같은 job에서 ...]" 같은 hint label 붙어있으면 제거 (LLM 판단 오염 방지)
    name = re.sub(r"\s*\[같은\s+job[^\]]*\]\s*$", "", name).strip()

    org_name = ""
    category_path = ""

    # candidate dict 직접 필드
    if c.get("org_name"):
        org_name = str(c["org_name"])
    if c.get("category_path"):
        category_path = str(c["category_path"])

    # raw에서 보강 (StatRecord 또는 dict)
    raw = c.get("raw")
    if raw is not None:
        # dataclass StatRecord
        if not org_name:
            _org = getattr(raw, "org_name", None)
            if _org:
                org_name = str(_org)
        if not category_path:
            _md = getattr(raw, "metadata", None)
            if isinstance(_md, dict):
                _cp = _md.get("category_path")
                if _cp:
                    category_path = str(_cp)
        # dict raw (예: from_job_success)
        if isinstance(raw, dict):
            if not org_name and raw.get("org_name"):
                org_name = str(raw["org_name"])
            if not category_path and raw.get("category_path"):
                category_path = str(raw["category_path"])

    return {
        "id": cid,
        "name": name,
        "org_name": org_name,
        "category_path": category_path,
    }


def _build_prompt(
    *,
    claim_text: str,
    indicator: str,
    population: str,
    time_period: str,
    parent_path: str,
    candidates: list[dict],
) -> str:
    """LLM 프롬프트 구성. candidates는 _normalize_candidate_for_prompt 결과."""

    cands_lines: list[str] = []
    for i, c in enumerate(candidates, start=1):
        line = f"  [{i}] id={c['id']!r}\n      name={c['name']!r}"
        if c.get("org_name"):
            line += f"\n      org={c['org_name']!r}"
        if c.get("category_path"):
            line += f"\n      category_path={c['category_path']!r}"
        cands_lines.append(line)
    cands_block = "\n".join(cands_lines) if cands_lines else "  (없음)"

    return f"""당신은 KOSIS 통계표 ranking reviewer입니다. 사용자 claim 검증에 적합한
*표 후보 N개를 의미 매칭 정도로 비교*해 0~1 점수로 ranking 하세요.

[사용자 claim]
- 원문: {claim_text or '(없음)'}
- 검증 지표 (indicator): {indicator or '(없음)'}
- 대상 집단/지역 (population): {population or '(없음)'}
- 시점 (time_period): {time_period or '(없음)'}
- 카테고리 경로 (parent_path): {parent_path or '(없음)'}

[후보 표 N개]
{cands_block}

[판단 기준 — 중요]
1. **한정자 매칭**:
   - claim의 indicator에 *한정자*가 있으면 (예: "체외 충격파 쇄석술 장비", "MRI 장비")
     → 표 이름에 *그 한정자*가 직접/상위로 매칭되는 표가 *최우선* (예: "특수의료장비",
       "진단방사선 장비" 등 한정자 포함 표가 score ↑).
   - claim의 indicator에 한정자가 *없으면* (예: 단순 "의료장비 수", "인구")
     → 표도 *한정자 없는 일반 집합* 표가 우선 (한정자 박힌 표는 *부분집합*이라 score ↓).
2. **population 매칭**:
   - 표 이름이나 category_path에 population을 포함/매칭하면 score ↑.
   - "시도별/시군구별"처럼 지역 분할 표는 대부분의 population에 적합.
3. **외국/장래/다른 도메인**:
   - 해외 통계, 장래 추계·전망, 도메인 불일치(인구↔경제↔기상) → score 0.0~0.1.
4. **score 의미**:
   - 0.9~1.0: 한정자/도메인 모두 정확 매칭, 정답 확신
   - 0.5~0.8: 도메인 맞고 한정자 부분 매칭 (상위 집합 등 row 검색으로 회수 가능)
   - 0.2~0.4: 도메인 맞지만 한정자 mismatch 또는 너무 광범위/협소
   - 0.0~0.1: 무관한 표 — 거부 권장

[응답 형식 — JSON only, 모든 후보에 대해 작성]
{{
  "rankings": [
    {{"id": "<후보 id>", "score": 0.95, "reason": "<한 줄 이유>"}},
    ...
  ]
}}

* id는 반드시 입력 candidates의 id 그대로 사용. 누락된 표는 score=0.0으로 처리됨.
* score 순서로 정렬할 필요는 없음 (호출자가 정렬).
"""


def _parse(raw: str, valid_ids: set[str]) -> list[dict] | None:
    """LLM JSON 응답 파싱. valid_ids에 있는 id만 채택.

    Returns:
        [{id, score, reason}, ...] — score 높은 순 정렬.
        파싱 실패 시 None.
    """
    if not raw:
        return None
    # JSON 블록 추출 (코드펜스 / 앞뒤 텍스트 핸들링)
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        logger.debug(f"[catalog_ranker] JSON 블록 못 찾음: raw={raw[:200]!r}")
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        logger.debug(f"[catalog_ranker] JSON parse 실패: {e}, raw={raw[:200]!r}")
        return None
    rankings = data.get("rankings")
    if not isinstance(rankings, list):
        return None

    out: list[dict] = []
    seen_ids: set[str] = set()
    for r in rankings:
        if not isinstance(r, dict):
            continue
        rid = str(r.get("id") or "").strip()
        if not rid or rid not in valid_ids or rid in seen_ids:
            continue
        try:
            score = float(r.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(score, 1.0))
        reason = str(r.get("reason") or "")[:200]
        out.append({"id": rid, "score": score, "reason": reason})
        seen_ids.add(rid)

    # LLM이 누락한 id는 score=0.0으로 추가
    for vid in valid_ids:
        if vid not in seen_ids:
            out.append({"id": vid, "score": 0.0, "reason": "LLM 응답에서 누락 — 기본 0.0"})

    # score 순 정렬
    out.sort(key=lambda x: -x["score"])
    return out


async def rank_candidates(
    *,
    claim_text: str,
    indicator: str,
    population: str,
    time_period: str,
    parent_path: str,
    candidates: list[dict],
    config: dict | None = None,
) -> list[dict] | None:
    """LLM batch ranking으로 후보 표 N개를 의미 점수로 정렬.

    Args:
        claim_text: claim 원문 (1~3문장).
        indicator: schema.indicator.
        population: schema.population.
        time_period: schema.time_period.
        parent_path: schema.parent_path.
        candidates: [{id, name, score, raw?}, ...] — catalog_search 결과.
        config: 전체 config dict. data_sources.kosis.catalog_ranker 섹션 사용.

    Returns:
        [{id, score, reason}, ...] — score 높은 순 정렬.
        candidates 비었거나 LLM 실패 시 None.
    """
    if not candidates:
        return None

    # 후보 정규화 (메타데이터 추출)
    norm_cands = [_normalize_candidate_for_prompt(c) for c in candidates]
    valid_ids: set[str] = {c["id"] for c in norm_cands if c["id"]}
    if not valid_ids:
        return None

    # config 추출 — data_sources.kosis.catalog_ranker
    _cfg = ((config or {}).get("data_sources") or {}).get("kosis") or {}
    _rk = _cfg.get("catalog_ranker") or {}
    model_tier = str(_rk.get("model_tier") or _DEFAULT_MODEL_TIER).strip().lower()

    prompt = _build_prompt(
        claim_text=claim_text,
        indicator=indicator,
        population=population,
        time_period=time_period,
        parent_path=parent_path,
        candidates=norm_cands,
    )

    from structverify.utils.llm_client import LLMClient
    llm = LLMClient(config=(config or {}).get("llm") or {})
    try:
        raw = await llm.generate(
            prompt=prompt,
            system_prompt="KOSIS 표 ranking reviewer. JSON만 응답.",
            model_tier=model_tier,
        )
    except Exception as e:
        logger.warning(f"[catalog_ranker] LLM 호출 실패: {e}")
        return None

    parsed = _parse(raw, valid_ids)
    if parsed is None:
        logger.warning(f"[catalog_ranker] 응답 파싱 실패 — raw={raw[:300]!r}")
        return None

    return parsed
