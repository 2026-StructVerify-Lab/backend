"""Render eval report.json as a single summary PNG."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _rate(block: dict[str, Any], *keys: str, default: float | None = None) -> float | None:
    cur: Any = block
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return float(cur) if isinstance(cur, (int, float)) else default


def _schema_strict(components: dict[str, Any]) -> float | None:
    sch = components.get("schema") or {}
    if isinstance(sch.get("strict"), dict):
        return _rate(sch, "strict", "accuracy")
    basis = sch.get("accuracy_basis")
    acc = sch.get("accuracy")
    if basis == "strict" or sch.get("strict"):
        return float(acc) if isinstance(acc, (int, float)) else None
    return _rate(sch, "strict", "accuracy", default=acc if basis != "aligned" else None)


def _schema_aligned(components: dict[str, Any]) -> float | None:
    sch = components.get("schema") or {}
    if isinstance(sch.get("aligned"), dict):
        return _rate(sch, "aligned", "accuracy")
    acc = sch.get("accuracy")
    if sch.get("accuracy_basis") == "aligned" or (
        sch.get("strict") and isinstance(acc, (int, float)) and acc > _rate(sch, "strict", "accuracy", default=0)
    ):
        return float(acc) if isinstance(acc, (int, float)) else None
    return _rate(sch, "aligned", "accuracy")


def _verdict_strict(components: dict[str, Any]) -> float | None:
    v = components.get("verdict") or {}
    if isinstance(v.get("strict"), dict):
        return _rate(v, "strict", "accuracy")
    if v.get("accuracy_basis") == "strict":
        return _rate(v, "accuracy")
    return _rate(v, "strict", "accuracy", default=_rate(v, "accuracy"))


def _verdict_aligned(components: dict[str, Any]) -> float | None:
    v = components.get("verdict") or {}
    if isinstance(v.get("aligned"), dict):
        return _rate(v, "aligned", "accuracy")
    if v.get("accuracy_basis") == "aligned":
        return _rate(v, "accuracy")
    return _rate(v, "aligned", "accuracy")


def _outcome_mode_block(outcome: dict[str, Any], mode: str) -> dict[str, Any]:
    block = outcome.get(mode)
    if isinstance(block, dict) and "verdict_accuracy" in block:
        return block
    if mode == outcome.get("primary_schema_mode", "oracle"):
        return outcome
    return {}


def collect_metrics(report: dict[str, Any]) -> list[tuple[str, float | None, str]]:
    """Label, value in [0,1], group name."""
    out: list[tuple[str, float | None, str]] = []
    outcome = report.get("outcome") or {}
    audit = report.get("audit") or {}
    comp = report.get("components") or {}

    primary = outcome.get("primary_schema_mode", "oracle")
    oracle_blk = _outcome_mode_block(outcome, "oracle")
    induce_blk = _outcome_mode_block(outcome, "induce")
    if oracle_blk or induce_blk:
        out.append(
            (
                "Outcome · verdict (oracle)",
                _rate(oracle_blk, "verdict_accuracy"),
                "primary",
            )
        )
        out.append(
            (
                "Outcome · verdict (induce)",
                _rate(induce_blk, "verdict_accuracy"),
                "primary",
            )
        )
        out.append(
            (
                "Outcome · value tol. (oracle)",
                _rate(oracle_blk, "value_tolerance_rate"),
                "primary",
            )
        )
        out.append(
            (
                "Outcome · value OK / verdict NG (oracle)",
                _rate(oracle_blk, "value_ok_verdict_wrong_rate"),
                "primary",
            )
        )
    else:
        out.append(("Outcome · verdict", _rate(outcome, "verdict_accuracy"), "primary"))
        out.append(
            ("Outcome · value tol.", _rate(outcome, "value_tolerance_rate"), "primary")
        )
    _ = primary  # used implicitly via blocks above
    out.append(("Audit · KOSIS grounding", _rate(audit, "kosis_grounding_rate"), "primary"))
    cv = _rate(audit, "constraint_violation_rate")
    out.append(
        (
            "Audit · constraints OK",
            (1.0 - cv) if cv is not None else None,
            "primary",
        )
    )

    det = comp.get("detection") or {}
    out.append(("Detection · F1", _rate(det, "f1"), "components"))
    out.append(("Detection · recall", _rate(det, "recall"), "components"))
    out.append(("Retrieval · accuracy", _rate(comp.get("retrieval") or {}, "accuracy"), "components"))
    out.append(("Schema · strict (3/3)", _schema_strict(comp), "components"))
    out.append(("Schema · aligned (2/3)", _schema_aligned(comp), "components"))
    out.append(("Verdict · strict", _verdict_strict(comp), "components"))
    out.append(("Verdict · aligned", _verdict_aligned(comp), "components"))
    return out


def find_latest_report(runs_root: Path) -> Path | None:
    candidates = sorted(runs_root.glob("eval_*/report.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def render_report_card(
    report: dict[str, Any],
    output_path: Path,
    *,
    dpi: int = 120,
) -> Path:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for report cards: pip install matplotlib"
        ) from e

    metrics = collect_metrics(report)
    labels = [m[0] for m in metrics]
    values = [m[1] if m[1] is not None else 0.0 for m in metrics]
    groups = [m[2] for m in metrics]
    colors = []
    for g in groups:
        if g == "primary":
            colors.append("#2563eb")
        else:
            colors.append("#64748b")

    oc = report.get("outcome") or {}
    primary_mode = oc.get("primary_schema_mode", "oracle")
    primary_blk = _outcome_mode_block(oc, str(primary_mode))
    outcome_n = primary_blk.get("n", oc.get("n", "?"))
    run_id = report.get("run_id", "unknown")
    dataset_id = report.get("dataset_id", "")

    fig_h = max(6.0, 0.35 * len(labels) + 2.5)
    fig, ax = plt.subplots(figsize=(10, fig_h), facecolor="#f8fafc")
    ax.set_facecolor("#f8fafc")

    y = list(range(len(labels)))
    bars = ax.barh(y, values, color=colors, height=0.65, edgecolor="white", linewidth=0.8)

    for i, (bar, val) in enumerate(zip(bars, [m[1] for m in metrics])):
        if val is None:
            txt = "n/a"
            bar.set_width(0)
        else:
            txt = f"{val * 100:.1f}%"
        ax.text(
            min(val or 0, 1.0) + 0.02 if val is not None else 0.02,
            bar.get_y() + bar.get_height() / 2,
            txt,
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold" if groups[i] == "primary" else "normal",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Score (higher is better)", fontsize=10)
    ax.axvline(1.0, color="#cbd5e1", linewidth=0.8, linestyle="--")
    ax.invert_yaxis()
    ax.set_title(
        f"StructVerify Eval · {dataset_id}\n{run_id}  (outcome n={outcome_n})",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    legend = [
        Patch(facecolor="#2563eb", label="Outcome / Audit (E2E)"),
        Patch(facecolor="#64748b", label="Components (fixtures)"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def load_report(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
