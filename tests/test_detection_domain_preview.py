"""tests/test_detection_domain_preview.py — domain preview unit tests."""
from structverify.core.schemas import BlockType, SIRBlock, SIRDocument, SourceType
from structverify.detection.domain.preview import _build_text_preview


def _sir_with_blocks() -> SIRDocument:
    return SIRDocument(
        source_type=SourceType.TEXT,
        blocks=[
            SIRBlock(
                block_id="b1",
                type=BlockType.HEADING,
                content="2024년 출생 통계",
            ),
            SIRBlock(
                block_id="b2",
                type=BlockType.PARAGRAPH,
                content="출생아 수는 2만 171명으로 전년 대비 감소했다.",
            ),
            SIRBlock(
                block_id="b3",
                type=BlockType.TABLE,
                content="표 데이터는 미리보기에서 제외",
            ),
        ],
    )


def test_build_text_preview_prefers_heading_then_paragraph():
    preview = _build_text_preview(_sir_with_blocks(), max_chars=200)
    assert "2024년 출생 통계" in preview
    assert "2만 171명" in preview
    assert "표 데이터" not in preview
