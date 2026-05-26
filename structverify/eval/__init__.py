"""
structverify.eval — evaluation dataset builder + pipeline harness.

Subpackages:
  builder/  — KOSIS-first golden eval set generation (EvalSetBuilder)
  harness/  — run VerificationPipeline on frozen datasets (EvalHarness)
"""

from structverify.eval.builder import (
    EvalArticle,
    EvalClaim,
    EvalManifest,
    EvalSetBuilder,
)
from structverify.eval.harness import EvalHarness, load_golden_dataset

__all__ = [
    "EvalArticle",
    "EvalClaim",
    "EvalHarness",
    "EvalManifest",
    "EvalSetBuilder",
    "load_golden_dataset",
]
