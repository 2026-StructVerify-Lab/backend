"""
tests/test_pipeline.py — 파이프라인 기본 테스트
"""
import sys
from structverify.preprocessing.segmenter import split_sentences, NUMERIC_PATTERN
from structverify.preprocessing.sir_builder import build_sir
from structverify.core.schemas import SourceType, VerdictType
from structverify.verification.verifier import verify_claim
from structverify.core.schemas import Claim, ClaimSchema, Evidence, SourceOffset
from uuid import uuid4


def test_numeric_detection():
    assert NUMERIC_PATTERN.search("고령화 비율은 64.2%에 이른다")
    assert NUMERIC_PATTERN.search("재배면적이 10만4943 ha")
    assert not NUMERIC_PATTERN.search("경제 상황이 악화되었다")


def test_sentence_split():
    text = "농가 수는 166,558가구이다. 고령화 비율은 64.2%이다."
    sents = split_sentences(text)
    assert len(sents) >= 2
    assert any(s.has_numeric for s in sents)
    assert all(s.graph_anchor_id for s in sents)  # v2: anchor 확인


def test_sir_builder():
    text = "제목입니다.\n\n농가 고령화 비율은 64.2%에 이른다.\n\n다른 문단."
    doc = build_sir(text, SourceType.TEXT)
    assert len(doc.blocks) == 3
    assert doc.blocks[0].graph_anchor_ids  # v2: anchor 확인


def test_verify_match():
    claim = Claim(doc_id=uuid4(), block_id="b0", sent_id="s0",
                  claim_text="고령화 64.2%",
                  schema=ClaimSchema(value=64.2),
                  source_offset=SourceOffset())
    evidence = Evidence(source_name="KOSIS", official_value=64.2)
    result = verify_claim(claim, evidence)
    assert result.verdict == VerdictType.MATCH


def test_verify_mismatch():
    claim = Claim(doc_id=uuid4(), block_id="b0", sent_id="s0",
                  claim_text="실업률 10%",
                  schema=ClaimSchema(value=10.0),
                  source_offset=SourceOffset())
    evidence = Evidence(source_name="KOSIS", official_value=7.2)
    result = verify_claim(claim, evidence)
    assert result.verdict == VerdictType.MISMATCH


def test_verify_unverifiable():
    claim = Claim(doc_id=uuid4(), block_id="b0", sent_id="s0",
                  claim_text="경제 악화",
                  source_offset=SourceOffset())
    result = verify_claim(claim, None)
    assert result.verdict == VerdictType.UNVERIFIABLE
