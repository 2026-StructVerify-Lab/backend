"""Render eval harness report.json as PNG charts and HTML summary."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from structverify.eval.harness.report_io import load_report
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def render_charts_and_html(run_dir: Path, report: dict[str, Any] | None = None) -> bool:
    """Write charts/*.png and report.html. Returns False if matplotlib missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning(
            "matplotlib not installed — skip charts. "
            "Install with: pip install -e '.[eval]'"
        )
        return False

    report = report or load_report(run_dir / "report.json")
    charts_dir = run_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    summary = report.get("summary", {})
    labels = report.get("confusion_matrix", {}).get("labels", [])
    matrix = report.get("confusion_matrix", {}).get("matrix", [])

    # verdict_accuracy.png — summary + per-class F1
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Verdict metrics", fontsize=12)
    axes[0].bar(
        ["accuracy", "extraction", "stat_id"],
        [
            summary.get("verdict_accuracy", 0),
            summary.get("extraction_recall", 0),
            summary.get("stat_id_accuracy", 0),
        ],
        color=["#4C78A8", "#72B7B2", "#F58518"],
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("rate")
    for i, v in enumerate(
        [
            summary.get("verdict_accuracy", 0),
            summary.get("extraction_recall", 0),
            summary.get("stat_id_accuracy", 0),
        ]
    ):
        axes[0].text(i, v + 0.02, _pct(v), ha="center", fontsize=9)

    per_v = report.get("per_verdict", {})
    cls_names = list(per_v.keys())
    f1s = [per_v[c].get("f1", 0) for c in cls_names]
    axes[1].bar(cls_names, f1s, color="#54A24B")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("F1 by verdict class")
    axes[1].set_ylabel("F1")
    fig.tight_layout()
    fig.savefig(charts_dir / "verdict_accuracy.png", dpi=120)
    plt.close(fig)

    # confusion_matrix.png
    if matrix and labels:
        fig, ax = plt.subplots(figsize=(5, 4))
        arr = np.array(matrix, dtype=float)
        im = ax.imshow(arr, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("gold")
        ax.set_title("Confusion matrix")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, int(arr[i, j]), ha="center", va="center", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(charts_dir / "confusion_matrix.png", dpi=120)
        plt.close(fig)

    # per_domain.png
    per_domain = report.get("per_domain", {})
    if per_domain:
        domains = list(per_domain.keys())
        accs = [per_domain[d].get("accuracy", 0) for d in domains]
        fig, ax = plt.subplots(figsize=(8, max(4, len(domains) * 0.35)))
        y = range(len(domains))
        ax.barh(y, accs, color="#4C78A8")
        ax.set_yticks(y, domains, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("verdict accuracy (matched claims)")
        ax.set_title("Per domain")
        fig.tight_layout()
        fig.savefig(charts_dir / "per_domain.png", dpi=120)
        plt.close(fig)

    # extraction_recall.png
    if per_domain:
        domains = list(per_domain.keys())
        recalls = [per_domain[d].get("extraction_recall", 0) for d in domains]
        fig, ax = plt.subplots(figsize=(8, max(4, len(domains) * 0.35)))
        y = range(len(domains))
        ax.barh(y, recalls, color="#72B7B2")
        ax.set_yticks(y, domains, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("extraction recall")
        ax.set_title("Claim extraction recall by domain")
        fig.tight_layout()
        fig.savefig(charts_dir / "extraction_recall.png", dpi=120)
        plt.close(fig)

    # per_article.png
    per_article = report.get("per_article", [])
    if per_article:
        ids = [a.get("article_id", "")[-8:] for a in per_article]
        accs = [a.get("accuracy", 0) for a in per_article]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(len(ids)), accs, color="#B279A2")
        ax.set_xticks(range(len(ids)), ids, rotation=90, fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_ylabel("accuracy")
        ax.set_title("Per article (matched claims)")
        fig.tight_layout()
        fig.savefig(charts_dir / "per_article.png", dpi=120)
        plt.close(fig)

    _write_html(run_dir, report)
    return True


def _write_html(run_dir: Path, report: dict[str, Any]) -> None:
    summary = report.get("summary", {})
    run_id = html.escape(str(report.get("run_id", "")))
    dataset_id = html.escape(str(report.get("dataset_id", "")))
    charts = [
        "verdict_accuracy.png",
        "confusion_matrix.png",
        "per_domain.png",
        "extraction_recall.png",
        "per_article.png",
    ]
    img_blocks = []
    for name in charts:
        if (run_dir / "charts" / name).exists():
            img_blocks.append(
                f'<section><h2>{html.escape(name.replace("_", " ").replace(".png", ""))}</h2>'
                f'<img src="charts/{name}" alt="{html.escape(name)}"/></section>'
            )

    body = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <title>Eval report — {run_id}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 1100px; }}
    .cards {{ display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }}
    .card {{ background: #f4f4f5; padding: 1rem 1.25rem; border-radius: 8px; min-width: 140px; }}
    .card strong {{ font-size: 1.5rem; display: block; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
    section {{ margin: 2rem 0; }}
  </style>
</head>
<body>
  <h1>StructVerify eval — {dataset_id}</h1>
  <p>Run: <code>{run_id}</code></p>
  <div class="cards">
    <div class="card"><span>Verdict accuracy</span><strong>{_pct(summary.get("verdict_accuracy", 0))}</strong></div>
    <div class="card"><span>Extraction recall</span><strong>{_pct(summary.get("extraction_recall", 0))}</strong></div>
    <div class="card"><span>Stat ID accuracy</span><strong>{_pct(summary.get("stat_id_accuracy", 0))}</strong></div>
    <div class="card"><span>Articles</span><strong>{report.get("articles_total", 0)}</strong></div>
    <div class="card"><span>Gold claims</span><strong>{report.get("claims_gold", 0)}</strong></div>
  </div>
  {"".join(img_blocks)}
</body>
</html>
"""
    (run_dir / "report.html").write_text(body, encoding="utf-8")


def render_eval_report(run_dir: str | Path) -> bool:
    """Public entry: render charts + HTML for an existing run directory."""
    return render_charts_and_html(Path(run_dir))
