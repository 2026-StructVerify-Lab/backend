"""
sv_platform.loaders.workspace_reader — agent_workspace 디렉토리에서
claim별 plan/trace 정보 읽어 프론트 응답용으로 정리.

agent_workspace 구조:
  agent_workspace/job_{JOB_ID}/
    meta.json
    source.txt
    successful_stat_ids.json
    verified_facts.json
    claims/
      {claim_id}/
        claim.json
        plan.json              ← Plan (claim_type, formula, initial_steps)
        verdict.json
        memory.md
        observations/
          iter_01_catalog_search.json   ← Reflect trace (action, input, output, summary)
          iter_02_fetch_evidence.json
          ...

이 모듈은 stateless reader — 파일 시스템 접근만. agent loop이 워크스페이스 위치를
바꾸면 여기도 따라가야 함 (현재 cwd 기준 agent_workspace/).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


# agent loop이 사용하는 workspace base 디렉토리.
# 환경변수 STRUCTVERIFY_AGENT_WORKSPACE로 override 가능.
DEFAULT_WORKSPACE_BASE = Path(
    os.environ.get(
        "STRUCTVERIFY_AGENT_WORKSPACE",
        "agent_workspace",
    )
)

# `iter_{NN}_{action}.json` 또는 `iter{NNN}_{action}.json` 둘 다 매칭.
# (agent_loop이 정식 저장하는 `iter_NN_X` 우선, 도구가 따로 저장하는 `iterNNN_X`는 폴백)
_ITER_FILENAME_RE = re.compile(r"^iter_?(\d{2,3})_(.+)\.json$")


def _safe_load_json(path: Path) -> dict | None:
    """JSON 안전 로드 — 파일 없거나 파싱 실패 시 None."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return None


def _find_job_dir(job_id: str, base: Path = DEFAULT_WORKSPACE_BASE) -> Path | None:
    """agent_workspace에서 해당 job_id 디렉토리 찾기.

    agent loop이 job_id를 변환해 디렉토리명을 만들기 때문에 정확 매칭이 아닐 수
    있음. job_{job_id} 또는 job_{job_id의 일부} 같이 prefix 매칭으로 탐색.
    """
    if not base.exists():
        return None
    # 정확 매칭 우선
    exact = base / f"job_{job_id}"
    if exact.exists():
        return exact
    # prefix 매칭 (job_id가 변환된 경우)
    for child in base.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("job_"):
            continue
        # 디렉토리명에 job_id의 앞 8자가 들어있으면 매칭
        if job_id[:8] in child.name:
            return child
    return None


def _summarize_plan(plan_dict: dict) -> dict | None:
    """plan.json 원본에서 프론트에 보일 필드만 추림.

    무거운 fallback 등은 생략. claim_type / required_data / steps / formula 중심.
    """
    if not isinstance(plan_dict, dict):
        return None
    out: dict[str, Any] = {
        "claim_type": plan_dict.get("claim_type"),
        "calculation_formula": plan_dict.get("calculation_formula"),
    }
    # required_data 경량화
    rd = plan_dict.get("required_data") or []
    if isinstance(rd, list):
        out["required_data"] = [
            {
                "indicator": d.get("indicator") if isinstance(d, dict) else None,
                "time": d.get("time") if isinstance(d, dict) else None,
                "population": d.get("population") if isinstance(d, dict) else None,
                "unit_hint": d.get("unit_hint") if isinstance(d, dict) else None,
            }
            for d in rd
            if isinstance(d, dict)
        ]
    # initial_steps — action + input + rationale 만
    steps = plan_dict.get("initial_steps") or []
    if isinstance(steps, list):
        out["initial_steps"] = [
            {
                "action": s.get("action") if isinstance(s, dict) else None,
                "input": s.get("input") if isinstance(s, dict) else None,
                "rationale": s.get("rationale") if isinstance(s, dict) else None,
            }
            for s in steps
            if isinstance(s, dict)
        ]
    return out


def _summarize_observation(obs: dict) -> dict:
    """iter observation에서 프론트 친화 필드만 추림.

    output 전체는 너무 무거움 (catalog candidates의 raw StatRecord 등).
    핵심 정보만:
      - iter, action, rationale, summary, success, error
      - output에서 가장 유의미한 짧은 신호 (candidates 이름 등) — 상위 3개만
    """
    iter_num = obs.get("iter_num") or obs.get("iter")
    action = obs.get("action")
    inp = obs.get("input")
    out = obs.get("output") or {}
    summary = obs.get("summary")
    success = obs.get("success", True)
    error = obs.get("error")

    # action별로 가장 유의미한 output 신호만 추림
    output_signal: dict[str, Any] = {}
    if isinstance(out, dict):
        if action in ("catalog_search", "explore_catalog"):
            # 후보 표 / 카테고리만 짧게
            cands = out.get("candidates") or []
            if isinstance(cands, list) and cands:
                output_signal["candidates_top3"] = [
                    {"id": c.get("id"), "name": c.get("name")}
                    for c in cands[:3]
                    if isinstance(c, dict)
                ]
            cats = out.get("categories") or []
            if isinstance(cats, list) and cats:
                output_signal["categories_top3"] = [
                    {
                        "category_label": c.get("category_label"),
                        "table_count": c.get("table_count"),
                    }
                    for c in cats[:3]
                    if isinstance(c, dict)
                ]
        elif action == "fetch_evidence":
            ev = out.get("evidence") or {}
            if isinstance(ev, dict):
                output_signal["evidence"] = {
                    "stat_table_id": ev.get("stat_table_id"),
                    "value": ev.get("value"),
                    "unit": ev.get("unit"),
                    "time_period": ev.get("time_period"),
                }
        elif action == "calculate":
            output_signal["result"] = out.get("result")
        elif action == "finish":
            output_signal["verdict"] = out.get("verdict")
            output_signal["confidence"] = out.get("confidence")

    return {
        "iter": iter_num,
        "action": action,
        "rationale": (inp or {}).get("rationale") if isinstance(inp, dict) else None,
        "input": _shrink_input(inp) if isinstance(inp, dict) else None,
        "summary": (summary or "")[:400] if isinstance(summary, str) else summary,
        "success": success,
        "error": error,
        "output": output_signal or None,
    }


def _shrink_input(inp: dict) -> dict:
    """input dict에서 핵심 키만 추림 (LLM이 큰 텍스트 넣는 경우 방지)."""
    KEEP = ("query", "category", "candidate_id", "params", "expression", "verdict",
            "indicator", "time_period")
    out: dict[str, Any] = {}
    for k in KEEP:
        if k in inp:
            v = inp[k]
            if isinstance(v, str) and len(v) > 200:
                v = v[:200] + "..."
            out[k] = v
    return out


def read_claim_workspace(
    job_id: str,
    claim_id: str,
    base: Path = DEFAULT_WORKSPACE_BASE,
) -> dict[str, Any]:
    """하나의 claim에 대해 plan + trace를 읽어 dict로 반환.

    파일 없으면 빈 dict ({plan: None, trace: []}).
    """
    job_dir = _find_job_dir(job_id, base)
    result: dict[str, Any] = {"plan": None, "trace": []}
    if job_dir is None:
        return result

    claim_dir = job_dir / "claims" / claim_id
    if not claim_dir.exists():
        return result

    # plan
    plan_data = _safe_load_json(claim_dir / "plan.json")
    if plan_data:
        result["plan"] = _summarize_plan(plan_data)

    # trace (observations 디렉토리)
    obs_dir = claim_dir / "observations"
    if not obs_dir.exists():
        return result

    # iter_{NN}_*.json 우선 사용. 동일 iter에 두 종류 파일이 있으면 (iter_01, iter001)
    # `iter_{NN}` 형태를 우선해서 dedup.
    iter_files: dict[int, Path] = {}
    iter_files_fallback: dict[int, Path] = {}
    for fp in obs_dir.iterdir():
        m = _ITER_FILENAME_RE.match(fp.name)
        if not m:
            continue
        n = int(m.group(1))
        if fp.name.startswith("iter_"):
            iter_files[n] = fp
        else:
            iter_files_fallback.setdefault(n, fp)
    # 빠진 iter는 fallback에서 채움
    for n, fp in iter_files_fallback.items():
        iter_files.setdefault(n, fp)

    trace: list[dict] = []
    for n in sorted(iter_files.keys()):
        obs = _safe_load_json(iter_files[n])
        if obs is None:
            continue
        trace.append(_summarize_observation(obs))
    result["trace"] = trace

    return result


def read_job_workspace_for_claims(
    job_id: str,
    claim_ids: list[str],
    base: Path = DEFAULT_WORKSPACE_BASE,
) -> dict[str, dict]:
    """여러 claim에 대해 한 번에 plan/trace를 읽음.

    Returns: {claim_id: {"plan": ..., "trace": ...}}
    """
    out: dict[str, dict] = {}
    for cid in claim_ids:
        try:
            out[cid] = read_claim_workspace(job_id, cid, base)
        except Exception as e:
            logger.debug(f"[workspace_reader] claim {cid} 읽기 실패: {e}")
            out[cid] = {"plan": None, "trace": []}
    return out
