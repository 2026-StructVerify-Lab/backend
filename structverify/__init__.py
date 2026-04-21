"""StructVerify Lab v2.0 — 도메인 독립형 LLM 기반 사실검증 플랫폼

Graph + JSON 하이브리드 저장 · 2-Agent 아키텍처 · 도메인 적응형 학습 루프
"""
from structverify.core.pipeline import verify_text, VerificationPipeline

__all__ = ["verify_text", "VerificationPipeline"]
__version__ = "0.2.0"
