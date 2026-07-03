"""StructVerify Lab v2.0 — 도메인 독립형 LLM 기반 사실검증 플랫폼

Graph + JSON 하이브리드 저장 · 2-Agent 아키텍처 · 도메인 적응형 학습 루프

[DONE] 김예슬
- 외부 사용자에게 보여줄 진입점 먼저 고정하고, 내부 모듈은 점진적으로 구현
- 구현 내용
verify_text()
verify_document()
내부에서 VerificationPipeline 호출
- 확인 사항
1. import 가능 여부
2. verify_text() 함수 실행 시 VerificationPipeline.run() 호출 여부

"""
from structverify.core.pipeline import verify_text, VerificationPipeline

__all__ = ["verify_text", "VerificationPipeline"]
__version__ = "0.2.0"
