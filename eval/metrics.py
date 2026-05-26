"""
eval.metrics — gold vs actual 비교로 단계별 메트릭 계산.

각 claim마다 다음 dict 반환:
  {
    "article_id", "claim_id", "claim_text",
    "gold_verdict", "actual_verdict", "verdict_correct",
    "gold_stat_id", "actual_stat_id", "stat_id_match",
    "gold_value", "actual_value", "value_match",
    "gold_indicator", "actual_indicator", "indicator_partial_match",
    "schema_value_match", "schema_time_match", "schema_pop_match",
    "elapsed_sec", "failure_mode" (if applicable),
  }
"""
from __future__ import annotations

import re
from typing import Any


def _normalize_indicator(s: str | None) -> str:
    """공백/조사/괄호 제거 정규화."""
    if not s:
        return ""
    s = re.sub(r"[\s（）()]+", "", str(s))
    s = re.sub(r"별$|율$|수$", "", s)
    return s.lower()


def _values_close(a: Any, b: Any, rel_tol: float = 0.05) -> bool:
    """5% 이내면 같음. None은 mismatch."""
    if a is None or b is None:
        return a == b
    try:
        af, bf = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if bf == 0.0:
        return abs(af) < 1e-6
    return abs(af - bf) / max(abs(bf), 1e-9) < rel_tol


def _classify_failure(actual_verdict: str, gold_verdict: str,
                      actual_stat_id: str | None, gold_stat_id: str | None,
                      actual_value: Any, gold_value: Any) -> str | None:
    """unverifiable/mismatch 시 *왜 실패했나* 분류.

    A. data_missing — gold도 unverifiable (정상)
    B. wrong_table — actual_stat_id != gold_stat_id
    C. no_table_picked — actual_stat_id is None
    D. wrong_value — table 맞는데 값 다름
    E. correct — match인데 시스템도 match (None 반환)
    """
    if actual_verdict == gold_verdict:
        return None  # 정확
    if gold_verdict == "unverifiable" and actual_verdict == "unverifiable":
        return "data_missing"
    if actual_verdict == "unverifiable":
        if actual_stat_id is None:
            return "no_table_picked"
        if gold_stat_id and actual_stat_id != gold_stat_id:
            return "wrong_table"
        return "row_match_failed"
    # verdict 자체가 다른 경우
    if actual_stat_id and gold_stat_id and actual_stat_id != gold_stat_id:
        return "wrong_table"
    if not _values_close(actual_value, gold_value):
        return "wrong_value"
    return "verdict_mismatch"


def compare_claim(gold: dict, actual: dict, elapsed_sec: float | None = None) -> dict:
    """gold claim + actual (claim+result) → 메트릭 row.

    Args:
        gold: articles.jsonl의 claim dict.
        actual: {"claim": Claim object dump, "result": VerificationResult dump}
                두 항목 모두 dict.
    """
    g_verdict = (gold.get("gold_verdict") or "").lower()
    g_schema = gold.get("gold_schema") or {}
    g_stat = gold.get("gold_stat_id")
    g_value = gold.get("gold_official_value")
    g_indicator = g_schema.get("indicator")

    a_claim = actual.get("claim") or {}
    a_result = actual.get("result") or {}

    a_verdict = str(a_result.get("verdict") or "").lower()
    a_schema = a_claim.get("schema") or {}
    a_evidence = a_result.get("evidence") or {}
    a_stat = a_evidence.get("stat_table_id")
    a_value = a_evidence.get("official_value")
    # derived (calculate) 사용 시 computed_value를 actual_value로
    if a_result.get("computed_value") is not None:
        a_value = a_result["computed_value"]
    a_indicator = a_schema.get("indicator")

    # 기본 매칭
    verdict_correct = a_verdict == g_verdict
    stat_id_match = (
        bool(a_stat) and bool(g_stat) and a_stat == g_stat
    )
    value_match = _values_close(a_value, g_value)

    # schema 단계
    ind_norm_g = _normalize_indicator(g_indicator)
    ind_norm_a = _normalize_indicator(a_indicator)
    indicator_partial_match = (
        bool(ind_norm_g) and bool(ind_norm_a)
        and (ind_norm_g in ind_norm_a or ind_norm_a in ind_norm_g)
    )
    schema_value_match = _values_close(
        a_schema.get("value"), g_schema.get("value"),
    )
    schema_time_match = (
        str(a_schema.get("time_period") or "").strip() ==
        str(g_schema.get("time_period") or "").strip()
    )
    schema_pop_match = (
        str(a_schema.get("population") or "").strip() ==
        str(g_schema.get("population") or "").strip()
    )

    failure_mode = _classify_failure(
        a_verdict, g_verdict, a_stat, g_stat, a_value, g_value,
    )

    return {
        "claim_id": gold.get("claim_id"),
        "claim_text": gold.get("claim_text"),
        "gold_verdict": g_verdict,
        "actual_verdict": a_verdict,
        "verdict_correct": verdict_correct,
        "gold_stat_id": g_stat,
        "actual_stat_id": a_stat,
        "stat_id_match": stat_id_match,
        "gold_value": g_value,
        "actual_value": a_value,
        "value_match": value_match,
        "gold_indicator": g_indicator,
        "actual_indicator": a_indicator,
        "indicator_partial_match": indicator_partial_match,
        "schema_value_match": schema_value_match,
        "schema_time_match": schema_time_match,
        "schema_pop_match": schema_pop_match,
        "elapsed_sec": elapsed_sec,
        "failure_mode": failure_mode,
    }


def _macro_f1(rows: list[dict], labels: list[str]) -> dict:
    """Macro F1 — fact-checking 논문 표준 (MultiFC, FEVER).

    각 label마다 P/R/F1 계산 후 단순 평균.
    Returns:
        {"per_class": {label: {p, r, f1, support}}, "macro": {p, r, f1}}
    """
    per_class: dict[str, dict] = {}
    p_list, r_list, f_list = [], [], []
    for lab in labels:
        tp = sum(1 for r in rows
                 if r.get("gold_verdict") == lab and r.get("actual_verdict") == lab)
        fp = sum(1 for r in rows
                 if r.get("gold_verdict") != lab and r.get("actual_verdict") == lab)
        fn = sum(1 for r in rows
                 if r.get("gold_verdict") == lab and r.get("actual_verdict") != lab)
        sup = sum(1 for r in rows if r.get("gold_verdict") == lab)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r_ = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r_ / (p + r_) if (p + r_) else 0.0
        per_class[lab] = {"precision": p, "recall": r_, "f1": f, "support": sup}
        p_list.append(p); r_list.append(r_); f_list.append(f)
    n = max(1, len(labels))
    return {
        "per_class": per_class,
        "macro": {
            "precision": sum(p_list) / n,
            "recall": sum(r_list) / n,
            "f1": sum(f_list) / n,
        },
    }


def _fever_score(rows: list[dict]) -> dict:
    """FEVER Score (Thorne et al. 2018) — strict:
        verdict 맞고 + evidence(stat_id) 맞고 + value ±5%이면 1, else 0.

    NEI/unverifiable의 경우 evidence 비교 X (verdict만으로 OK).
    """
    n_total = len(rows)
    n_strict = 0
    n_label_only = 0
    for r in rows:
        verdict_ok = r.get("verdict_correct", False)
        if not verdict_ok:
            continue
        n_label_only += 1
        gv = r.get("gold_verdict")
        # unverifiable은 evidence 비교 안 함
        if gv == "unverifiable":
            n_strict += 1
            continue
        # match/mismatch는 evidence(stat_id + value) 일치도 필요
        if r.get("stat_id_match") and r.get("value_match"):
            n_strict += 1
    return {
        "fever_score": n_strict / n_total if n_total else 0.0,
        "label_only_acc": n_label_only / n_total if n_total else 0.0,
        "n_total": n_total,
        "n_strict": n_strict,
        "n_label_only": n_label_only,
    }


def aggregate(rows: list[dict]) -> dict:
    """전체 row → 요약 메트릭. error 행도 안전하게 처리 (.get default 0).

    포함 metrics (fact verification 논문 표준):
      - Label Accuracy (FEVER, MultiFC)
      - FEVER Score — strict label + evidence (Thorne et al. 2018)
      - Macro F1 (MultiFC, TabFact)
      - per-class P/R/F1
      - Slot accuracy (ACE/DocRED 류)
      - Mismatch precision/recall (misinformation detection)
    """
    if not rows:
        return {"total": 0}

    n = len(rows)
    n_verdict_correct = sum(1 for r in rows if r.get("verdict_correct"))
    n_stat_match = sum(1 for r in rows if r.get("stat_id_match"))
    n_val_match = sum(1 for r in rows if r.get("value_match"))
    n_ind_partial = sum(1 for r in rows if r.get("indicator_partial_match"))
    n_schema_val = sum(1 for r in rows if r.get("schema_value_match"))
    n_schema_time = sum(1 for r in rows if r.get("schema_time_match"))
    n_schema_pop = sum(1 for r in rows if r.get("schema_pop_match"))

    # verdict 단위 confusion
    confusion: dict[tuple, int] = {}
    for r in rows:
        key = (r.get("gold_verdict", "?"), r.get("actual_verdict", "?"))
        confusion[key] = confusion.get(key, 0) + 1

    # precision/recall (mismatch 기준 — misinformation detection 능력)
    tp = sum(1 for r in rows
             if r.get("gold_verdict") == "mismatch" and r.get("actual_verdict") == "mismatch")
    fp = sum(1 for r in rows
             if r.get("gold_verdict") != "mismatch" and r.get("actual_verdict") == "mismatch")
    fn = sum(1 for r in rows
             if r.get("gold_verdict") == "mismatch" and r.get("actual_verdict") != "mismatch")
    precision_mm = tp / (tp + fp) if (tp + fp) else 0.0
    recall_mm = tp / (tp + fn) if (tp + fn) else 0.0

    # failure mode 분포
    fmode: dict[str, int] = {}
    for r in rows:
        m = r.get("failure_mode")
        if m:
            fmode[m] = fmode.get(m, 0) + 1

    # latency
    elapsed = [r.get("elapsed_sec") for r in rows if r.get("elapsed_sec") is not None]
    avg_elapsed = sum(elapsed) / len(elapsed) if elapsed else None

    # ── 논문 표준 metrics ────────────────────────────────────────
    labels = ["match", "mismatch", "unverifiable"]
    macro = _macro_f1(rows, labels)
    fever = _fever_score(rows)

    return {
        "total_claims": n,
        # FEVER-style (논문 main metrics)
        "label_accuracy": n_verdict_correct / n,        # = Label Accuracy
        "fever_score": fever["fever_score"],            # FEVER Score (strict)
        "macro_f1": macro["macro"]["f1"],
        "macro_precision": macro["macro"]["precision"],
        "macro_recall": macro["macro"]["recall"],
        "per_class": macro["per_class"],
        # 호환용 (기존 보고서 key 유지)
        "verdict_accuracy": n_verdict_correct / n,
        # Evidence (FEVER score 분해)
        "stat_id_accuracy": n_stat_match / n,
        "value_accuracy": n_val_match / n,
        # Schema slot (ACE/DocRED 스타일)
        "indicator_partial_accuracy": n_ind_partial / n,
        "schema_value_accuracy": n_schema_val / n,
        "schema_time_accuracy": n_schema_time / n,
        "schema_pop_accuracy": n_schema_pop / n,
        # Mismatch detection (misinformation 잡기)
        "mismatch_precision": precision_mm,
        "mismatch_recall": recall_mm,
        # Diagnostics
        "confusion_matrix": {
            f"{g}→{a}": c for (g, a), c in sorted(confusion.items())
        },
        "failure_modes": fmode,
        "avg_elapsed_sec": avg_elapsed,
        "n_with_elapsed": len(elapsed),
    }
