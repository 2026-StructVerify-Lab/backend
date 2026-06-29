"""#68-① Library public entrypoints == VerificationPipeline.run (smoke, mocked).

API 키·KOSIS·LLM 없이 진입점이 pipeline.run 과 동일 경로인지만 확인한다.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from structverify.core.pipeline import (
    VerificationEngine,
    VerificationPipeline,
    verify_document,
    verify_text,
)
from structverify.core.schemas import SIRDocument, SourceType, VerificationReport

pytestmark = pytest.mark.asyncio


def _fake_report() -> VerificationReport:
    return VerificationReport(
        document=SIRDocument(source_type=SourceType.TEXT, source_uri="smoke"),
    )


@pytest.fixture
def fake_report() -> VerificationReport:
    return _fake_report()


@pytest.fixture
def cfg() -> dict:
    return {"app": {"debug": True}}


async def test_verify_text_delegates_to_pipeline_run(
    fake_report: VerificationReport, cfg: dict,
) -> None:
    with patch.object(VerificationPipeline, "run", new_callable=AsyncMock) as run:
        run.return_value = fake_report
        out = await verify_text("테스트 문장입니다.", config=cfg)
        # verify_text → run(text, "text"); source_text는 run() 기본값 사용(미전달)
        run.assert_awaited_once_with("테스트 문장입니다.", "text")
        assert out is fake_report


async def test_verify_document_delegates_to_pipeline_run(
    fake_report: VerificationReport, cfg: dict,
) -> None:
    with patch.object(VerificationPipeline, "run", new_callable=AsyncMock) as run:
        run.return_value = fake_report
        out = await verify_document(
            "https://example.com/article",
            source_type="url",
            config=cfg,
            source_text="추출된 본문",
        )
        run.assert_awaited_once_with(
            "https://example.com/article", "url", "추출된 본문",
        )
        assert out is fake_report


async def test_verification_engine_verify_text_delegates_to_pipeline_run(
    fake_report: VerificationReport, cfg: dict,
) -> None:
    with patch.object(VerificationPipeline, "run", new_callable=AsyncMock) as run:
        run.return_value = fake_report
        engine = VerificationEngine(cfg)
        out = await engine.verify_text("동일 입력")
        run.assert_awaited_once_with("동일 입력", "text")
        assert out is fake_report


async def test_verification_engine_verify_document_delegates_to_pipeline_run(
    fake_report: VerificationReport, cfg: dict,
) -> None:
    with patch.object(VerificationPipeline, "run", new_callable=AsyncMock) as run:
        run.return_value = fake_report
        engine = VerificationEngine(cfg)
        out = await engine.verify_document(
            "report.pdf", source_type="pdf", source_text="pdf 본문",
        )
        run.assert_awaited_once_with("report.pdf", "pdf", "pdf 본문")
        assert out is fake_report


async def test_verify_text_equals_direct_pipeline_run(
    fake_report: VerificationReport, cfg: dict,
) -> None:
    """verify_text(text, config) == VerificationPipeline(config).run(text, 'text')."""
    with patch.object(VerificationPipeline, "run", new_callable=AsyncMock) as run:
        run.return_value = fake_report
        text = "2023년 전국 고용률은 62.6%이다."
        from_api = await verify_text(text, config=cfg)
        from_pipeline = await VerificationPipeline(cfg).run(text, "text")
        assert from_api is from_pipeline
        assert run.await_count == 2


async def test_verify_document_equals_direct_pipeline_run(
    fake_report: VerificationReport, cfg: dict,
) -> None:
    with patch.object(VerificationPipeline, "run", new_callable=AsyncMock) as run:
        run.return_value = fake_report
        source, stype, body = "doc.txt", "text", "본문 텍스트"
        from_api = await verify_document(
            source, source_type=stype, config=cfg, source_text=body,
        )
        from_pipeline = await VerificationPipeline(cfg).run(source, stype, body)
        assert from_api is from_pipeline
        assert run.await_count == 2
