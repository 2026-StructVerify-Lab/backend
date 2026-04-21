from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from structverify.core.config_loader import load_config
from structverify.core.pipeline import VerificationPipeline
from tools.common import save_json, serialize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run StructVerify pipeline and optionally save output JSON.")
    parser.add_argument("--text", type=str, help="Inline text input")
    parser.add_argument("--input-file", type=str, help="Path to file containing input text")
    parser.add_argument("--source-type", type=str, default="text", choices=["text", "url", "pdf", "docx"])
    parser.add_argument("--config", type=str, default=os.getenv("SV_CONFIG_PATH", "config/default.yaml"))
    parser.add_argument("--output-name", type=str, default="pipeline_report")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    source: str
    if args.text:
        source = args.text
    elif args.input_file:
        source = Path(args.input_file).read_text(encoding="utf-8") if args.source_type == "text" else args.input_file
    else:
        raise SystemExit("--text 또는 --input-file 중 하나는 필요합니다.")

    if args.no_save:
        os.environ["SV_SAVE_RESULTS"] = "0"

    config = load_config(args.config)
    pipeline = VerificationPipeline(config)
    report = await pipeline.run(source, args.source_type)

    output = serialize(report)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    saved = save_json(output, args.output_name, metadata={"source_type": args.source_type, "config": args.config})
    if saved:
        print(f"\n[saved] {saved}")


if __name__ == "__main__":
    asyncio.run(main())
