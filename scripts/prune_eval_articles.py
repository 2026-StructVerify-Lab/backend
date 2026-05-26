#!/usr/bin/env python3
"""Remove eval articles by id and rebuild eval/builder/.build_state.json from JSONL."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from structverify.eval.builder.dataset_writer import DatasetWriter
from structverify.eval.builder.schemas import BuildState, EvalManifest
from structverify.eval.builder.validator import EvalArticleValidator

STATE_PATH = Path("eval/builder/.build_state.json")


def _max_article_seq(article_ids: list[str]) -> int:
    best = 0
    for aid in article_ids:
        m = re.search(r"_(\d+)$", aid)
        if m:
            best = max(best, int(m.group(1)))
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune eval articles and sync build state")
    parser.add_argument(
        "--dataset-id",
        default="structverify_eval_v4",
        help="Dataset id under eval/datasets/",
    )
    parser.add_argument(
        "--remove",
        nargs="*",
        default=[],
        help="article_id values to remove (e.g. eval_policy_0037)",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Rebuild .build_state.json + manifest from current articles.jsonl (no deletions)",
    )
    parser.add_argument(
        "--config",
        default="config/eval_builder.yaml",
        help="eval_builder.yaml for manifest hash",
    )
    args = parser.parse_args()

    import yaml

    with open(args.config, encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f) or {}

    remove_set = set(args.remove)
    if not args.sync_only and not remove_set:
        print("Provide --remove ID(s) or use --sync-only")
        return 1

    writer = DatasetWriter(Path("eval/datasets"), args.dataset_id)
    articles = writer.load_articles()
    kept = [a for a in articles if a.article_id not in remove_set]
    removed = len(articles) - len(kept)
    if not args.sync_only and removed == 0:
        print(f"No articles matched --remove ids in {writer.articles_path}")
        print("  (Those IDs are not in the file — maybe already deleted.)")
        print("  If JSONL is already correct, run: --sync-only")
        return 1

    writer.ensure_dirs()
    if removed:
        with open(writer.articles_path, "w", encoding="utf-8") as f:
            for art in kept:
                f.write(json.dumps(art.model_dump(mode="json"), ensure_ascii=False) + "\n")

    mode = eval_cfg.get("mode", "pilot")
    seed = int(eval_cfg.get("seed", 42))
    state = BuildState(dataset_id=args.dataset_id, mode=mode, seed=seed)
    for art in kept:
        EvalArticleValidator.register_article_facts(art, state)
        EvalArticleValidator.update_counts(art, state)
    state.next_article_seq = _max_article_seq([a.article_id for a in kept]) + 1

    writer.write_build_state(STATE_PATH, state)

    manifest = EvalManifest(
        dataset_id=args.dataset_id,
        mode=mode,
        article_count=state.articles_written,
        claim_count=state.claims_written,
        status="building",
    )
    writer.write_manifest(manifest, eval_cfg, status="building")

    if removed:
        print(f"Removed {removed} article(s): {sorted(remove_set)}")
    elif args.sync_only:
        print("Sync-only: articles.jsonl unchanged")
    print(f"Remaining: {len(kept)} articles, {state.claims_written} claims")
    print(f"next_article_seq={state.next_article_seq}")
    if removed:
        print(f"Updated {writer.articles_path}")
    print(f"Updated {STATE_PATH} (status=building)")
    print(
        "\nNext: python scripts/build_eval_set.py "
        f"--config {args.config} --resume"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
