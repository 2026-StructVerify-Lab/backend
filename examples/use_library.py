"""structverify 라이브러리 사용 예시.

설치:
    pip install -e .            # (+ pip install openai  — upstage/openai provider 쓸 때)

환경변수 (.env 또는 export):
    NCP_API_KEY=...                 # HCX 사용 시 (LLM·임베딩·reranker 공용)
    UPSTAGE_API_KEY=up-...          # Upstage(Solar) LLM 사용 시
    OPENAI_API_KEY=sk-...           # OpenAI provider 사용 시
    KOSIS_API_KEY=...               # KOSIS 데이터 소스
    PGVECTOR_DSN=postgresql://...   # 카탈로그(pgvector)
    CUSTOM_DB_DSN=...               # custom_db 데이터 소스 사용 시

provider 전환:
    config/default.yaml 의 `llm.provider` 를 hcx | openai | upstage 로 변경.
    (예시는 config/default_example.yaml 참고)

실행:
    python examples/use_library.py
"""
import asyncio

import structverify


# ── 1) 가장 간단한 텍스트 검증 (config 생략 시 config/default.yaml 사용) ──
async def example_text():
    report = await structverify.verify_text(
        "올해 4월 출생아 수는 총 2만 717명으로 지난해 같은 달보다 8.7% 늘었다."
    )
    for r in report.results:
        official = getattr(getattr(r, "evidence", None), "official_value", None)
        print(r.verdict, "| 공식값:", official, "|", r.explanation)


# ── 2) 문서(URL/PDF/DOCX) 검증 — verify_document ──
async def example_document():
    report = await structverify.verify_document(
        "https://example.com/article",
        source_type="url",      # "url" | "pdf" | "docx" | "text"
    )
    print(report.verdict_distribution)


# ── 3) config 주입으로 provider/데이터소스 바꾸기 (예: HCX 키 없이 Upstage) ──
#   주의: provider 만 바꾸려면 config/default.yaml 의 llm.provider 를 수정하는 게 안전합니다.
#   (아래는 dict 주입 형태 — 전체 설정은 config/default.yaml 을 따릅니다.)
async def example_with_config():
    config = {
        "llm": {
            "provider": "upstage",
            "api_key_env": "UPSTAGE_API_KEY",
            "base_url": "https://api.upstage.ai/v1",
            "models": {"heavy": "solar-pro", "light": "solar-mini", "structured": "solar-pro"},
        },
    }
    report = await structverify.verify_text("검증할 문장", config=config)
    print(report)


if __name__ == "__main__":
    # 공개 API 확인
    print("공개 API:", structverify.__all__)
    # 실제 검증 (LLM/DB 키가 세팅돼 있어야 동작)
    asyncio.run(example_text())
