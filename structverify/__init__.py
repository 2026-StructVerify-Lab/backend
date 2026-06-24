"""StructVerify Lab v2.0 — 도메인 독립형 LLM 기반 사실검증 플랫폼

Graph + JSON 하이브리드 저장 · 2-Agent 아키텍처 · 도메인 적응형 학습 루프

공개 API:
    verify_text(text, config)                      — 텍스트 일회성 검증
    verify_document(source, source_type, config)   — URL/PDF/DOCX/TEXT 일회성 검증
    VerificationEngine(config)                     — config를 주입해 두고 재사용 (객체형)
    VerificationPipeline(config)                   — 저수준 파이프라인

[DONE] 김예슬
- 외부 사용자에게 보여줄 진입점 먼저 고정하고, 내부 모듈은 점진적으로 구현
- verify_text() / verify_document() → 내부에서 VerificationPipeline 호출
- VerificationEngine: 같은 config로 여러 번 검증하는 객체형 진입점
- 확인 사항: (1) import 가능 (2) verify_text() 실행 시 VerificationPipeline.run() 호출

[v2 - 김예슬] 지연 로딩(PEP 562):
    `import structverify` 시점에는 무거운 pipeline/deps(trafilatura·httpx·bs4 등)를
    끌어오지 않는다. 실제로 verify_text 등을 *사용*할 때 처음 로드된다.
    → import/패키지 탐지가 가볍고, 선택적 deps 없이도 import 자체는 성공.
"""

__all__ = [
    "verify_text",
    "verify_document",
    "VerificationEngine",
    "VerificationPipeline",
]
__version__ = "0.2.0"


def __getattr__(name: str):
    # PEP 562 — 공개 심볼을 *사용 시점* 에 core.pipeline 에서 지연 로딩.
    if name in __all__:
        from structverify.core import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
