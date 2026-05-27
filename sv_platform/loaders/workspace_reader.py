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

_WS_RE = re.compile(r"\s+")


def _normalize_ws(s: str) -> str:
    """공백/개행/탭을 단일 스페이스로 정규화 후 strip."""
    return _WS_RE.sub(" ", s).strip()


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


def _find_job_dir(
    job_id: str,
    base: Path = DEFAULT_WORKSPACE_BASE,
    source_text: str | None = None,
) -> Path | None:
    """agent_workspace에서 해당 job_id 디렉토리 찾기.

    매칭 우선순위:
      1. 정확 매칭: agent_workspace/job_{job_id}/
      2. source_text 일치: 같은 source_text를 가진 가장 최근 워크스페이스
         (라이브러리가 sv_platform job_id가 아닌 자체 doc_id로 디렉토리를
          만들기 때문에 sv_platform job_id로 정확 매칭이 거의 안 됨.
          같은 텍스트로 검증한 워크스페이스라면 같은 잡으로 간주.)
      3. prefix 매칭: 디렉토리명에 job_id의 앞 8자 (안전망 — 위험할 수 있어
         source_text가 None일 때만 시도)
    """
    if not base.exists():
        return None
    # 1. 정확 매칭
    exact = base / f"job_{job_id}"
    if exact.exists():
        return exact

    # 2. source_text 일치 매칭 — 라이브러리가 자체 doc_id로 디렉토리를
    #    만들었을 때의 안전한 fallback.
    #
    # ★ Whitespace 정규화 비교: 라이브러리가 source 저장 시 newline을 공백으로
    # 압축해 single-line으로 쓰는 반면 sv_platform 입력엔 `\n\n` 등이 그대로
    # 있어 strict equality는 거의 실패함. 모든 연속 whitespace를 단일 공백으로
    # 치환한 후 비교.
    #
    # ★ 같은 source_text로 여러 워크스페이스(과거 잡 + 새 잡)가 있을 수 있음.
    # source_text 일치 워크스페이스를 모두 모은 뒤, 그 안의 claims/ 디렉토리
    # mtime을 비교해 가장 최근 활동(=현재 진행 중 잡) 선택.
    if source_text:
        target = _normalize_ws(source_text)
        try:
            dirs = [
                p for p in base.iterdir()
                if p.is_dir() and p.name.startswith("job_")
            ]
        except OSError:
            dirs = []
        candidates: list[tuple[float, Path]] = []
        for d in dirs:
            src_file = d / "source.txt"
            if not src_file.exists():
                continue
            try:
                with src_file.open("r", encoding="utf-8") as f:
                    if _normalize_ws(f.read()) != target:
                        continue
            except OSError:
                continue
            # claims/ 디렉토리 mtime — 새 잡이면 방금 만들어진 claim이 들어있음
            claims_dir = d / "claims"
            if claims_dir.exists():
                try:
                    mt = claims_dir.stat().st_mtime
                except OSError:
                    mt = 0.0
            else:
                # claims/ 없으면 부모 mtime으로 차순위 fallback
                try:
                    mt = d.stat().st_mtime
                except OSError:
                    mt = 0.0
            candidates.append((mt, d))
        if candidates:
            # 가장 최근 활동 워크스페이스
            candidates.sort(key=lambda t: t[0], reverse=True)
            return candidates[0][1]

    # 3. prefix 매칭 — source_text가 없을 때만 (다른 잡 데이터 끌어올 위험)
    if not source_text:
        for child in base.iterdir():
            if not child.is_dir():
                continue
            if not child.name.startswith("job_"):
                continue
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
      - iter, action, thought, rationale, summary, success, error
      - output에서 가장 유의미한 짧은 신호 (candidates 이름 등) — 상위 3개만

    [2026-05-27] LLM '생각' 노출:
      - thought (reflect LLM의 자연어 사고) 그대로 전달
      - confidence_so_far, proposed_verdict 동반
      - input/summary 잘림 한계 완화 (200→1000, 400→1500). 프론트가 카드에서
        펼쳤을 때 LLM이 실제로 어떤 정보를 보냈는지 사용자가 확인 가능.
    """
    iter_num = obs.get("iter_num") or obs.get("iter")
    action = obs.get("action")
    inp = obs.get("input")
    out = obs.get("output") or {}
    summary = obs.get("summary")
    success = obs.get("success", True)
    error = obs.get("error")
    # ↓ LLM 사고 — loop.py가 obs_dict에 박은 신규 필드 (없으면 None)
    thought = obs.get("thought")
    confidence_so_far = obs.get("confidence_so_far")
    proposed_verdict = obs.get("proposed_verdict")

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
        "summary": (summary or "")[:1500] if isinstance(summary, str) else summary,
        "success": success,
        "error": error,
        "output": output_signal or None,
        # ↓ [2026-05-27] LLM 자연어 사고 + 신뢰도 + 제안된 verdict
        "thought": thought if isinstance(thought, str) and thought else None,
        "confidence_so_far": confidence_so_far
            if isinstance(confidence_so_far, (int, float)) else None,
        "proposed_verdict": proposed_verdict
            if isinstance(proposed_verdict, str) and proposed_verdict else None,
    }


def _shrink_input(inp: dict) -> dict:
    """input dict에서 핵심 키만 추림 (LLM이 큰 텍스트 넣는 경우 방지).

    [2026-05-27] 값 truncate 한계 200→1000자로 완화. 카드에서 펼쳤을 때 LLM이
    어떤 query/params/expression을 실제로 보냈는지 확인할 수 있게.
    """
    KEEP = ("query", "category", "candidate_id", "params", "expression", "verdict",
            "indicator", "time_period", "formula", "vars", "explanation", "rationale")
    out: dict[str, Any] = {}
    for k in KEEP:
        if k in inp:
            v = inp[k]
            if isinstance(v, str) and len(v) > 1000:
                v = v[:1000] + "..."
            out[k] = v
    return out


def read_claim_workspace(
    job_id: str,
    claim_id: str,
    base: Path = DEFAULT_WORKSPACE_BASE,
    source_text: str | None = None,
) -> dict[str, Any]:
    """하나의 claim에 대해 plan + trace를 읽어 dict로 반환.

    파일 없으면 빈 dict ({plan: None, trace: []}).
    """
    job_dir = _find_job_dir(job_id, base, source_text=source_text)
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
    source_text: str | None = None,
) -> dict[str, dict]:
    """여러 claim에 대해 한 번에 plan/trace를 읽음.

    Returns: {claim_id: {"plan": ..., "trace": ...}}
    """
    out: dict[str, dict] = {}
    for cid in claim_ids:
        try:
            out[cid] = read_claim_workspace(job_id, cid, base, source_text=source_text)
        except Exception as e:
            logger.debug(f"[workspace_reader] claim {cid} 읽기 실패: {e}")
            out[cid] = {"plan": None, "trace": []}
    return out


def read_partial_job_workspace(
    job_id: str,
    base: Path = DEFAULT_WORKSPACE_BASE,
    source_text: str | None = None,
    created_after: float | None = None,
) -> dict[str, Any] | None:
    """실시간 폴링용 — 잡이 진행 중에도 workspace를 읽어 partial result 생성.

    agent_workspace 디렉토리를 직접 스캔해서:
      - 각 claim 디렉토리(claims/<cid>/)에서 claim.json + plan.json + trace + verdict.json
        (있는 만큼)을 읽어 합침.
      - 잡이 진행 중이라 verdict.json이 아직 없을 수도 있음 — 그래도 plan+trace만으로
        부분 진행 상황 노출.
      - claim 디렉토리 자체가 없으면 None 반환 (Step 7 이전).

    Returns dict 형태:
      {
        "claims": [
          {claim_id, claim_text, sent_id, schema, plan?, trace?, verdict?, ...},
          ...
        ]
      }
    또는 None (워크스페이스 없음).
    """
    job_dir = _find_job_dir(job_id, base, source_text=source_text)
    if job_dir is None:
        return None

    claims_dir = job_dir / "claims"
    if not claims_dir.exists() or not claims_dir.is_dir():
        return None

    partial_claims: list[dict[str, Any]] = []
    # 잡 안의 모든 claim 디렉토리 (mtime 정렬 — agent loop이 만든 순서)
    # workspace_dir 재사용 버그로 과거 잡들의 claim도 같은 디렉토리에 누적될 수
    # 있음. created_after 주어지면 그 시각 이후 mtime의 claim만 (현재 진행 잡).
    try:
        claim_subdirs = sorted(
            [p for p in claims_dir.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
        )
    except OSError:
        return None
    if created_after is not None:
        cutoff = float(created_after)
        claim_subdirs = [
            p for p in claim_subdirs
            if p.stat().st_mtime >= cutoff
        ]

    for cdir in claim_subdirs:
        cid = cdir.name
        merged: dict[str, Any] = {"claim_id": cid}

        # claim.json — sent_id, claim_text, schema 등 기본 정보
        claim_data = _safe_load_json(cdir / "claim.json")
        if claim_data:
            for k in ("doc_id", "block_id", "sent_id", "claim_text",
                      "claim_type", "schema"):
                if k in claim_data:
                    merged[k] = claim_data[k]

        # plan.json — Planner 결과 (있으면)
        plan_data = _safe_load_json(cdir / "plan.json")
        if plan_data:
            merged["plan"] = _summarize_plan(plan_data)

        # observations/iter_*.json — trace (있는 만큼)
        obs_dir = cdir / "observations"
        if obs_dir.exists():
            iter_files: dict[int, Path] = {}
            iter_files_fb: dict[int, Path] = {}
            for fp in obs_dir.iterdir():
                m = _ITER_FILENAME_RE.match(fp.name)
                if not m:
                    continue
                n = int(m.group(1))
                if fp.name.startswith("iter_"):
                    iter_files[n] = fp
                else:
                    iter_files_fb.setdefault(n, fp)
            for n, fp in iter_files_fb.items():
                iter_files.setdefault(n, fp)
            trace: list[dict] = []
            for n in sorted(iter_files.keys()):
                obs = _safe_load_json(iter_files[n])
                if obs is None:
                    continue
                trace.append(_summarize_observation(obs))
            if trace:
                merged["trace"] = trace

        # verdict.json — 잡이 진행 중이면 없을 수도 있음
        verdict_data = _safe_load_json(cdir / "verdict.json")
        if verdict_data:
            for k in ("verdict", "confidence", "explanation",
                      "iterations_used", "stop_reason"):
                if k in verdict_data:
                    merged[k] = verdict_data[k]

        partial_claims.append(merged)

    return {"claims": partial_claims} if partial_claims else None


# ── [2026-05-27] LLM raw trace reader (개발자 콘솔 탭용) ──────────────
# runtime_agent가 매 LLM 호출마다 observations/llm_traces/<name>.json에
# {name, ts, prompt, response, prompt_chars, response_chars}를 저장함.
# 이 함수는 잡 내 모든 claim의 llm_traces를 시간순으로 평탄화해 반환.

def read_job_llm_traces(
    job_id: str,
    base: Path = DEFAULT_WORKSPACE_BASE,
    source_text: str | None = None,
    created_after: float | None = None,
) -> list[dict[str, Any]]:
    """잡 내 모든 claim의 LLM raw trace를 평탄 리스트로.

    각 entry:
      {claim_id, name, ts, prompt, response, prompt_chars, response_chars, mtime}

    ts(ISO string) 기준 오름차순 정렬 — 콘솔에서 시간순 스트림처럼 보이게.
    """
    job_dir = _find_job_dir(job_id, base, source_text=source_text)
    if job_dir is None:
        return []

    claims_dir = job_dir / "claims"
    if not claims_dir.exists():
        return []

    out: list[dict[str, Any]] = []
    try:
        claim_subdirs = [p for p in claims_dir.iterdir() if p.is_dir()]
    except OSError:
        return []
    if created_after is not None:
        cutoff = float(created_after)
        claim_subdirs = [
            p for p in claim_subdirs
            if p.stat().st_mtime >= cutoff
        ]

    for cdir in claim_subdirs:
        traces_dir = cdir / "observations" / "llm_traces"
        if not traces_dir.exists():
            continue
        try:
            files = [p for p in traces_dir.iterdir() if p.suffix == ".json"]
        except OSError:
            continue
        for fp in files:
            data = _safe_load_json(fp)
            if not data:
                continue
            try:
                mt = fp.stat().st_mtime
            except OSError:
                mt = 0.0
            out.append({
                "claim_id": cdir.name,
                "name": data.get("name") or fp.stem,
                "ts": data.get("ts"),
                "prompt": data.get("prompt"),
                "response": data.get("response"),
                "prompt_chars": data.get("prompt_chars"),
                "response_chars": data.get("response_chars"),
                "mtime": mt,
            })

    # ts 우선, 없으면 mtime으로 정렬
    out.sort(key=lambda d: (d.get("ts") or "", d.get("mtime") or 0))
    return out
