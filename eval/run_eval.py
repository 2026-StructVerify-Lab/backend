"""
eval.run_eval — articles.jsonl 돌려서 verdict + 단계별 메트릭 산출.

사용법:
    python -m eval.run_eval                  # 전체
    python -m eval.run_eval --limit 10       # 처음 10개만
    python -m eval.run_eval --domain healthcare  # 특정 도메인만
    python -m eval.run_eval --resume         # 기존 results 디렉토리에 누락 article만

결과:
    eval/results/{timestamp}/
      ├── summary.md           # 사람 읽기용 요약
      ├── aggregate.json       # 요약 메트릭
      ├── rows.json            # claim별 row 메트릭
      ├── articles/            # article별 raw 결과 (재시작용)
      └── eval_run.log
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# 백엔드 root를 path에 추가 (eval 디렉토리에서 실행해도 import 가능)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from structverify.core.config_loader import load_config  # noqa: E402
from structverify.core.pipeline import VerificationPipeline  # noqa: E402

from eval.metrics import aggregate, compare_claim, enrich_with_semantic_similarity  # noqa: E402
from eval.report import compare_to_baseline, write_reports  # noqa: E402


ARTICLES_PATH = _ROOT / "articles.jsonl"


def _slim_claim(c: dict) -> dict:
    """report.claims의 한 entry를 *eval에 필요한 필드만* 추출 (순환 참조 회피)."""
    if not isinstance(c, dict):
        return {}
    sch = c.get("schema") or {}
    return {
        "claim_id": c.get("claim_id"),
        "claim_text": c.get("claim_text"),
        "sent_id": c.get("sent_id"),
        "schema": {
            "indicator": sch.get("indicator"),
            "value": sch.get("value"),
            "unit": sch.get("unit"),
            "time_period": sch.get("time_period"),
            "population": sch.get("population"),
            "value_role": sch.get("value_role"),
            "prev_value": sch.get("prev_value"),
            "prev_time_period": sch.get("prev_time_period"),
            "parent_path": sch.get("parent_path"),
        },
    }


def _safe_extract_report(report: Any) -> dict:
    """report 객체에서 *eval 필수 필드*만 attribute 접근으로 추출.

    report.model_dump(mode="json")가 graph 순환 참조로 실패하는 경우 대비.
    Claim/Result 객체는 graph_nodes 등 무거운 필드 X — 우리는 schema/evidence만 필요.
    """
    out: dict[str, list] = {"claims": [], "results": []}
    for c in (getattr(report, "claims", None) or []):
        try:
            sch = getattr(c, "schema", None)
            sch_d = {}
            if sch is not None:
                for k in ("indicator", "value", "unit", "time_period", "population",
                          "value_role", "prev_value", "prev_time_period", "parent_path"):
                    sch_d[k] = getattr(sch, k, None)
            out["claims"].append({
                "claim_id": str(getattr(c, "claim_id", "")),
                "claim_text": getattr(c, "claim_text", None),
                "sent_id": getattr(c, "sent_id", None),
                "schema": sch_d,
            })
        except Exception:
            continue
    for r in (getattr(report, "results", None) or []):
        try:
            ev = getattr(r, "evidence", None)
            ev_d = None
            if ev is not None:
                ev_d = {
                    "stat_table_id": getattr(ev, "stat_table_id", None),
                    "official_value": getattr(ev, "official_value", None),
                    "unit": getattr(ev, "unit", None),
                    "time_period": getattr(ev, "time_period", None),
                    "source_name": getattr(ev, "source_name", None),
                }
            verdict = getattr(r, "verdict", None)
            if verdict is not None and hasattr(verdict, "value"):
                verdict = verdict.value
            out["results"].append({
                "claim_id": str(getattr(r, "claim_id", "")),
                "verdict": verdict,
                "confidence": getattr(r, "confidence", None),
                "computed_value": getattr(r, "computed_value", None),
                "formula": getattr(r, "formula", None),
                "explanation": getattr(r, "explanation", None),
                "evidence": ev_d,
            })
        except Exception:
            continue
    return out


def _slim_result(r: dict) -> dict:
    """report.results의 한 entry를 *eval에 필요한 필드만*."""
    if not isinstance(r, dict):
        return {}
    ev = r.get("evidence") or {}
    return {
        "claim_id": r.get("claim_id"),
        "verdict": r.get("verdict"),
        "confidence": r.get("confidence"),
        "computed_value": r.get("computed_value"),
        "formula": r.get("formula"),
        "explanation": r.get("explanation"),
        "evidence": {
            "stat_table_id": ev.get("stat_table_id") if isinstance(ev, dict) else None,
            "official_value": ev.get("official_value") if isinstance(ev, dict) else None,
            "unit": ev.get("unit") if isinstance(ev, dict) else None,
            "time_period": ev.get("time_period") if isinstance(ev, dict) else None,
            "source_name": ev.get("source_name") if isinstance(ev, dict) else None,
        } if ev else None,
    }


def setup_file_logging(log_path: Path) -> None:
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s — %(message)s"
    ))
    root = logging.getLogger()
    root.addHandler(fh)


def load_articles(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"articles.jsonl not found at {path}")
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARN line {i}: parse fail: {e}", file=sys.stderr)
    return out


def match_actual_to_gold(article: dict, report_dump: dict) -> list[dict]:
    """report.claims + report.results → gold.claims와 claim_text 기반 매칭.

    Returns:
        list of {"gold": ..., "actual": {"claim": claim_dict, "result": result_dict}}
    """
    claims = report_dump.get("claims") or []
    results = report_dump.get("results") or []

    # claim_id → claim_dict, result_dict
    claim_by_id = {c.get("claim_id"): c for c in claims}
    result_by_claim_id = {r.get("claim_id"): r for r in results}

    # actual: claim_text → (claim_dict, result_dict)
    # ★ 한 claim_text가 여러 sub-claim으로 분기될 수 있음 (population/role별).
    # 매칭은 claim_text + (population, value_role) 키로 best-effort.
    actual_by_text: dict[str, list[dict]] = {}
    for c in claims:
        txt = (c.get("claim_text") or "").strip()
        rid = c.get("claim_id")
        actual_by_text.setdefault(txt, []).append({
            "claim": c, "result": result_by_claim_id.get(rid),
        })

    matches = []
    for gold_claim in article.get("claims", []):
        gtxt = (gold_claim.get("claim_text") or "").strip()
        candidates = actual_by_text.get(gtxt, [])
        if not candidates:
            # 매칭 못 함 — 시스템이 claim을 안 만듦
            matches.append({"gold": gold_claim, "actual": None})
            continue
        # 한 candidate면 그대로. 여러 개면 population 일치 우선.
        gpop = (gold_claim.get("gold_schema") or {}).get("population") or ""
        best = None
        for cand in candidates:
            cpop = ((cand["claim"] or {}).get("schema") or {}).get("population") or ""
            if cpop and gpop and cpop == gpop:
                best = cand
                break
        if best is None:
            best = candidates[0]
        matches.append({"gold": gold_claim, "actual": best})

    return matches


async def run_one_oracle_claim(
    pipeline: VerificationPipeline,
    article_id: str,
    gold_claim: dict,
    art_dir: Path,
) -> dict:
    """Setting A (Oracle): gold.claim_text를 *그대로 article처럼* 파이프라인에 투입.

    Detection 단계 영향 *최소화* — claim_text가 짧고 단일 문장이라
    detect_claims가 거의 그대로 1개로 추출 → schema/verify 단계 정상.

    Returns:
        eval row dict (compare_claim 결과 + article_id).
    """
    text = gold_claim.get("claim_text") or ""
    gid = gold_claim.get("claim_id", "?")
    t0 = time.time()
    if not text:
        return {
            "article_id": article_id, "claim_id": gid,
            "claim_text": text,
            "gold_verdict": (gold_claim.get("gold_verdict") or "").lower(),
            "actual_verdict": "no_input",
            "verdict_correct": False,
            "failure_mode": "no_claim_text",
            "elapsed_sec": 0.0,
            "mode": "oracle",
        }
    try:
        report = await pipeline.run(text, source_type="text")
    except Exception as e:
        logging.error(f"[eval-oracle] {gid}: pipeline 실패: {type(e).__name__}: {e}")
        return {
            "article_id": article_id, "claim_id": gid,
            "claim_text": text,
            "gold_verdict": (gold_claim.get("gold_verdict") or "").lower(),
            "actual_verdict": "error",
            "verdict_correct": False,
            "failure_mode": "pipeline_exception",
            "error": f"{type(e).__name__}: {e}",
            "elapsed_sec": time.time() - t0,
            "mode": "oracle",
        }
    elapsed = time.time() - t0
    report_dump = _safe_extract_report(report)

    claims = report_dump.get("claims") or []
    results_list = report_dump.get("results") or []
    result_by_cid = {r.get("claim_id"): r for r in results_list}

    if not claims:
        row = {
            "article_id": article_id, "claim_id": gid,
            "claim_text": text,
            "gold_verdict": (gold_claim.get("gold_verdict") or "").lower(),
            "actual_verdict": "no_claim_extracted",
            "verdict_correct": False,
            "failure_mode": "claim_not_extracted",
            "elapsed_sec": elapsed,
            "mode": "oracle",
        }
    else:
        # 단일 claim_text 입력 → 시스템이 1+개 sub-claim으로 분기 가능 (population별).
        # gold.population 일치하는 sub-claim을 best로.
        gold_pop = (gold_claim.get("gold_schema") or {}).get("population") or ""
        best = None
        for c in claims:
            cpop = ((c or {}).get("schema") or {}).get("population") or ""
            if gold_pop and cpop and gold_pop == cpop:
                best = c
                break
        if best is None:
            best = claims[0]
        actual = {
            "claim": best,
            "result": result_by_cid.get(best.get("claim_id")),
        }
        row = compare_claim(gold_claim, actual, elapsed_sec=elapsed)
        row["article_id"] = article_id
        row["mode"] = "oracle"

    # raw dump (옵션)
    try:
        oracle_dir = art_dir / "_oracle"
        oracle_dir.mkdir(parents=True, exist_ok=True)
        (oracle_dir / f"{gid}.json").write_text(
            json.dumps({
                "claim_id": gid,
                "elapsed_sec": elapsed,
                "report": {
                    "claims": [_slim_claim(c) for c in claims],
                    "results": [_slim_result(r) for r in results_list],
                },
            }, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:
        pass

    return row


async def run_one_article(
    pipeline: VerificationPipeline,
    article: dict,
    art_dir: Path,
) -> tuple[list[dict], dict | None]:
    """한 article 검증 → (메트릭 row 리스트, raw report dict)."""
    article_id = article.get("article_id", "?")
    text = article.get("article_text") or ""
    if not text:
        return [], None

    t0 = time.time()
    try:
        report = await pipeline.run(text, source_type="text")
    except Exception as e:
        logging.error(f"[eval] {article_id}: pipeline 실패 {type(e).__name__}: {e}")
        return [{
            "article_id": article_id,
            "claim_id": c.get("claim_id"),
            "claim_text": c.get("claim_text"),
            "gold_verdict": (c.get("gold_verdict") or "").lower(),
            "actual_verdict": "error",
            "verdict_correct": False,
            "failure_mode": "pipeline_exception",
            "error": f"{type(e).__name__}: {e}",
        } for c in article.get("claims", [])], None
    elapsed = time.time() - t0

    # report.model_dump(mode="json") 가 graph 순환 참조로 실패 — *직접 attribute 추출*
    report_dump = _safe_extract_report(report)

    # article-level raw 저장 (재현용) — *얇은 dump*만 (claims+results)
    # 전체 report에는 graph nodes 등 순환 참조 위험이 있어 명시 필드만 추출.
    art_path = art_dir / f"{article_id}.json"
    try:
        slim = {
            "claims": [
                _slim_claim(c) for c in (report_dump.get("claims") or [])
            ],
            "results": [
                _slim_result(r) for r in (report_dump.get("results") or [])
            ],
        }
        art_path.write_text(
            json.dumps({
                "article_id": article_id,
                "elapsed_sec": elapsed,
                "report": slim,
            }, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logging.warning(f"[eval] {article_id}: article raw 저장 실패: {e}")

    matches = match_actual_to_gold(article, report_dump)
    # claim 단위 elapsed는 article-level / N_claims 로 분배 (대략)
    n_claims = max(1, len(matches))
    per_claim_elapsed = elapsed / n_claims

    rows: list[dict] = []
    for m in matches:
        gold = m["gold"]
        actual = m["actual"]
        if actual is None:
            rows.append({
                "article_id": article_id,
                "claim_id": gold.get("claim_id"),
                "claim_text": gold.get("claim_text"),
                "gold_verdict": (gold.get("gold_verdict") or "").lower(),
                "actual_verdict": "no_claim_extracted",
                "verdict_correct": False,
                "failure_mode": "claim_not_extracted",
                "elapsed_sec": per_claim_elapsed,
            })
            continue
        row = compare_claim(gold, actual, elapsed_sec=per_claim_elapsed)
        row["article_id"] = article_id
        rows.append(row)
    return rows, report_dump


async def amain(args: argparse.Namespace) -> int:
    articles = load_articles(ARTICLES_PATH)

    # 필터
    if args.domain:
        articles = [a for a in articles if a.get("intended_domain") == args.domain]
    if args.limit and args.limit > 0:
        articles = articles[: args.limit]

    if not articles:
        print("처리할 article 없음", file=sys.stderr)
        return 1

    # 출력 디렉토리
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _ROOT / "eval" / "results" / ts
    art_dir = out_dir / "articles"
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(out_dir / "eval_run.log")

    print(f"[eval] articles to run: {len(articles)}")
    print(f"[eval] output: {out_dir}")

    config = load_config()
    # [2026-05-27] Oracle mode — claim detection LLM 필터링 건너뛰기.
    # config.eval.bypass_detection=true → runtime_agent가 모든 문장을 claim으로 변환.
    if args.mode == "oracle":
        config.setdefault("eval", {})["bypass_detection"] = True
        print("[oracle] bypass_detection=True — detect_claims LLM 필터링 SKIP")
    pipeline = VerificationPipeline(config)

    all_rows: list[dict] = []

    if args.mode == "oracle":
        # ── Setting A: Oracle claim ───────────────────────────────
        # 각 gold.claim_text를 article 입력으로 → detection 영향 최소화
        # 표준: FEVER (Thorne et al. 2018), SciFact (Wadden et al. 2020) 등에서
        # main result로 사용하는 setting
        flat_claims: list[tuple[str, dict]] = []
        for art in articles:
            aid = art.get("article_id", "?")
            for gold in art.get("claims", []) or []:
                flat_claims.append((aid, gold))
        if args.claim_limit and args.claim_limit > 0:
            flat_claims = flat_claims[: args.claim_limit]
        print(f"[oracle] total claims to run: {len(flat_claims)}")
        for seen, (aid, gold) in enumerate(flat_claims, 1):
            gid = gold.get("claim_id", "?")
            print(
                f"[oracle {seen}/{len(flat_claims)}] {gid}", flush=True,
            )
            row = await run_one_oracle_claim(pipeline, aid, gold, art_dir)
            all_rows.append(row)
            ok = sum(1 for r in all_rows if r.get("verdict_correct"))
            print(f"  → so far: correct {ok}/{len(all_rows)} ({ok/len(all_rows):.0%})")
    else:
        # ── Setting B: End-to-end ─────────────────────────────────
        for i, art in enumerate(articles, 1):
            aid = art.get("article_id", "?")
            print(f"[{i}/{len(articles)}] {aid} (domain={art.get('intended_domain')})", flush=True)
            try:
                rows, _report = await run_one_article(pipeline, art, art_dir)
            except Exception as e:
                print(f"  ! exception: {e}", file=sys.stderr)
                rows = [{
                    "article_id": aid,
                    "claim_id": c.get("claim_id"),
                    "claim_text": c.get("claim_text"),
                    "gold_verdict": (c.get("gold_verdict") or "").lower(),
                    "actual_verdict": "exception",
                    "verdict_correct": False,
                    "failure_mode": "exception",
                    "error": f"{type(e).__name__}: {e}",
                    "mode": "e2e",
                } for c in art.get("claims", [])]
            for r in rows:
                r.setdefault("mode", "e2e")
            all_rows.extend(rows)
            ok = sum(1 for r in all_rows if r.get("verdict_correct"))
            print(f"  → claims so far: {len(all_rows)}, correct: {ok} ({ok/len(all_rows):.0%})")

    # [Option C] semantic similarity 보강 (HCX embedding API)
    try:
        await enrich_with_semantic_similarity(all_rows, threshold=0.65)
        print("[eval] semantic similarity 계산 완료")
    except Exception as e:
        print(f"[eval] semantic 보강 실패 (무시): {e}")

    # 집계
    agg = aggregate(all_rows)
    write_reports(out_dir, all_rows, agg)
    print(f"\n[eval] DONE. Summary: {out_dir}/summary.md")
    print(f"  verdict_accuracy = {agg['verdict_accuracy']:.1%}")
    print(f"  stat_id_accuracy = {agg['stat_id_accuracy']:.1%}")
    print(f"  value_accuracy = {agg['value_accuracy']:.1%}")

    # baseline 비교 (있으면)
    baseline_path = _ROOT / "eval" / "baselines" / "latest.json"
    if baseline_path.exists() and not args.no_diff:
        try:
            base_rows = json.loads(baseline_path.read_text(encoding="utf-8"))
            diff = compare_to_baseline(all_rows, base_rows)
            (out_dir / "diff_vs_baseline.json").write_text(
                json.dumps(diff, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"\n[diff vs baseline] wins={len(diff['wins'])}, regressions={len(diff['regressions'])}")
        except Exception as e:
            print(f"  ! baseline 비교 실패: {e}")

    # 새 baseline 저장 (--save-baseline 옵션)
    if args.save_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(all_rows, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"  baseline 저장: {baseline_path}")

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="StructVerify evaluation harness")
    p.add_argument("--mode", choices=["oracle", "e2e"], default="oracle",
                   help="oracle = Setting A (gold claim_text 직접 입력, FEVER-style), "
                        "e2e = Setting B (article 전체 입력, detection 포함)")
    p.add_argument("--limit", type=int, default=0,
                   help="처리할 article 수 제한 (0이면 전체). oracle에서도 article 단위로 적용")
    p.add_argument("--domain", type=str, default=None,
                   help="특정 intended_domain만")
    p.add_argument("--claim-limit", type=int, default=0,
                   help="oracle 모드에서만 — 처리할 claim 수 상한 (0이면 전체)")
    p.add_argument("--no-diff", action="store_true",
                   help="baseline 비교 skip")
    p.add_argument("--save-baseline", action="store_true",
                   help="이번 결과를 baselines/latest.json에 저장")
    args = p.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
