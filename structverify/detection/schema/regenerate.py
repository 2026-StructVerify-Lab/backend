"""detection/schema/regenerate.py — observation 기반 schema 재분류.

schema_inductor.py에서 분리 (로직 move-only, 동작 변경 없음).

[2026-05-27] regenerate_schema — ReplanTool 연동
"""
from __future__ import annotations

import json
import re
from typing import Any

from structverify.detection.prompts.schema import REGENERATE_SCHEMA_PROMPT
from structverify.detection._config import model_tier_for
from structverify.detection._llm import get_llm_client
from structverify.detection.schema.validate import _safe_float
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


# ── [2026-05-27] regenerate_schema ───────────────────────────────────

def _summarize_observations_for_schema(observations: list[dict]) -> str:
    """observation list를 schema regeneration 프롬프트용으로 요약.

    fetch_evidence observation에서 추출:
      - 시도된 stat_id, 표 이름
      - PRD_DE 분포 (어떤 시점 데이터가 있는지)
      - ITM_NM/C2_NM unique 값 (어떤 분류가 있는지)
      - 매칭된 sample row의 indicator + value (있으면)
    """
    if not observations:
        return "(없음)"
    lines: list[str] = []
    for i, ob in enumerate(observations[:10], 1):  # 최대 10개
        if not isinstance(ob, dict):
            continue
        action = ob.get("action", "")
        summary = str(ob.get("summary", ""))[:160]
        success = ob.get("success")
        line = f"  [{i}] action={action} success={success}\n      summary={summary!r}"
        # fetch 관련 부가 정보
        if action == "fetch_evidence":
            stat_id = ob.get("stat_id")
            fv = ob.get("fetched_value")
            ft = ob.get("fetched_time")
            if stat_id:
                line += f"\n      stat_id={stat_id!r}"
            if fv is not None:
                line += f" fetched_value={fv} time={ft!r}"
            tried = ob.get("tried_candidates")
            if tried:
                line += f"\n      tried_candidates={tried}"
        elif action == "catalog_search":
            top3 = ob.get("candidates_top3")
            if top3:
                line += f"\n      top3={top3}"
        lines.append(line)
    return "\n".join(lines) if lines else "(없음)"


async def regenerate_schema(
    *,
    claim_text: str,
    original_schema: dict | None,
    observations: list[dict],
    config: dict | None = None,
) -> dict | None:
    """원문 + observation으로 schema 재분류.

    Args:
        claim_text: 원본 claim 문장.
        original_schema: 초기 induction이 만든 schema dict.
        observations: ReplanTool이 수집한 observation 요약 리스트.
        config: 전체 config.

    Returns:
        새 schema dict (value_role 등 갱신). 실패 시 None.
    """
    import json

    if not claim_text:
        logger.warning("[schema_inductor.regenerate] claim_text 비어있음")
        return None
    orig_json = "(없음)"
    if original_schema:
        try:
            orig_json = json.dumps(
                original_schema, ensure_ascii=False, indent=2, default=str,
            )
        except Exception:
            orig_json = str(original_schema)

    obs_summary = _summarize_observations_for_schema(observations or [])

    prompt = REGENERATE_SCHEMA_PROMPT.format(
        claim_text=claim_text,
        original_schema_json=orig_json,
        observations_summary=obs_summary,
    )
    logger.info(
        f"[schema_inductor.regenerate] prompt 구성 완료 ({len(prompt)}자) — "
        f"obs={len(observations or [])}건"
    )

    llm = get_llm_client(config)
    try:
        raw = await llm.generate(
            prompt=prompt,
            system_prompt="당신은 통계 검증 schema 수정자입니다. JSON만 응답.",
            model_tier=model_tier_for(config, "schema_regenerate"),
        )
    except Exception as e:
        logger.warning(f"[schema_inductor.regenerate] LLM 호출 실패: {e}")
        return None

    logger.info(
        f"[schema_inductor.regenerate] LLM 응답 본문 ↓\n"
        f"────── SCHEMA REGEN RESPONSE START ──────\n"
        f"{raw}\n"
        f"────── SCHEMA REGEN RESPONSE END ──────"
    )

    # JSON 추출
    m = re.search(r"\{[\s\S]*\}", raw or "")
    if not m:
        logger.warning("[schema_inductor.regenerate] JSON 블록 못 찾음")
        return None
    try:
        new_schema = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        logger.warning(f"[schema_inductor.regenerate] JSON parse 실패: {e}")
        return None
    if not isinstance(new_schema, dict):
        return None

    # 정규화 (필수 키 유지)
    out: dict[str, Any] = {}
    for k in [
        "indicator", "time_period", "unit", "population",
        "value", "value_role",
        "prev_value", "prev_time_period", "prev_phrase",
        "parent_path", "modifier",
    ]:
        if k in new_schema:
            out[k] = new_schema[k]
    # value/prev_value는 안전 float
    if "value" in out:
        out["value"] = _safe_float(out["value"])
    if "prev_value" in out:
        out["prev_value"] = _safe_float(out["prev_value"])

    reason = new_schema.get("reason") or ""
    logger.info(
        f"[schema_inductor.regenerate] 새 schema — "
        f"value_role={out.get('value_role')!r}, "
        f"prev_time_period={out.get('prev_time_period')!r}, "
        f"prev_value={out.get('prev_value')!r}, "
        f"reason={reason[:160]!r}"
    )
    return out
