"""
tests/test_step9_explainer.py — Step 9 explainer 테스트

실행:
    # API 키 없이 — 단위 테스트
    python -m pytest tests/test_step9_explainer.py -k "unit" -v

    # OpenAI 키 필요
    export OPENAI_API_KEY="sk-xxx"
    python -m pytest tests/test_step9_explainer.py -k "llm" -v -s
"""
import os
import pytest
from uuid import uuid4

HAS_LLM_KEY = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("NCP_API_KEY"))
skip_no_llm = pytest.mark.skipif(not HAS_LLM_KEY, reason="LLM API 키 없음")

@pytest.fixture
def llm_config():
    if os.environ.get("OPENAI_API_KEY"):
        return {
            "provider": "openai",
            "models": {"heavy": "gpt-4o", "light": "gpt-4o-mini"},
            "temperature": 0.1,
            "max_tokens": 512,
            "openai_key_env": "OPENAI_API_KEY",
        }
    return {
        "provider": "hcx",
        "models": {"heavy": "HCX-003", "light": "HCX-DASH-001"},
        "temperature": 0.1,
        "max_tokens": 512,
        "api_key_env": "NCP_API_KEY",
    }

@pytest.fixture
def sample_claim():
    from structverify.core.schemas import Claim, ClaimSchema, SourceOffset
    return Claim(
        doc_id=uuid4(),
        block_id="b0001",
        sent_id="s0001",
        claim_text="2023년 기준 65세 이상 과수 농가 경영주 비율이 64.2%에 달해 역대 최고치를 기록했다.",
        check_worthy_score=0.92,
        schema=ClaimSchema(
            indicator="65세 이상 경영주 비율",
            time_period="2023",
            unit="%",
            population="과수 농가",
            value=64.2,
        ),
    )


@pytest.fixture
def match_result(sample_claim):
    from structverify.core.schemas import (
        VerificationResult, VerdictType, Evidence, ProvenanceRecord
    )
    from datetime import datetime
    return VerificationResult(
        claim_id=sample_claim.claim_id,
        verdict=VerdictType.MATCH,
        confidence=0.95,
        evidence=Evidence(
            source_name="KOSIS 농림어업총조사",
            stat_table_id="DT_1EA1019",
            official_value=64.1,
            unit="%",
            time_period="2023",
            provenance=ProvenanceRecord(
                source_connector="kosis",
                source_id="DT_1EA1019",
                query_used="65세 이상 과수 농가",
                fetched_at=datetime(2026, 4, 24),
            ),
        ),
    )


@pytest.fixture
def mismatch_result(sample_claim):
    from structverify.core.schemas import (
        VerificationResult, VerdictType, MismatchType, Evidence, ProvenanceRecord
    )
    from datetime import datetime
    return VerificationResult(
        claim_id=sample_claim.claim_id,
        verdict=VerdictType.MISMATCH,
        confidence=0.88,
        mismatch_type=MismatchType.VALUE,
        evidence=Evidence(
            source_name="KOSIS 농림어업총조사",
            stat_table_id="DT_1EA1019",
            official_value=58.3,
            unit="%",
            time_period="2023",
            provenance=ProvenanceRecord(
                source_connector="kosis",
                source_id="DT_1EA1019",
                query_used="65세 이상 과수 농가",
                fetched_at=datetime(2026, 4, 24),
            ),
        ),
    )


@pytest.fixture
def unverifiable_result(sample_claim):
    from structverify.core.schemas import VerificationResult, VerdictType
    return VerificationResult(
        claim_id=sample_claim.claim_id,
        verdict=VerdictType.UNVERIFIABLE,
        confidence=0.2,
        evidence=None,
    )


# ── 단위 테스트 (API 키 불필요) ─────────────────────────────────────────

def test_unit_mismatch_reason_text():
    """_mismatch_reason_text: MismatchType별 설명 문구 확인"""
    from structverify.explanation.explainer import _mismatch_reason_text
    from structverify.core.schemas import MismatchType

    assert "시점" in _mismatch_reason_text(MismatchType.TIME_PERIOD)
    assert "과장" in _mismatch_reason_text(MismatchType.EXAGGERATION)
    assert "집단" in _mismatch_reason_text(MismatchType.POPULATION)
    assert "수치" in _mismatch_reason_text(MismatchType.VALUE)
    assert _mismatch_reason_text(None) == "수치 불일치"


def test_unit_calc_diff_pct():
    """_calc_diff_pct: 차이 비율 계산"""
    from structverify.explanation.explainer import _calc_diff_pct
    assert _calc_diff_pct(64.2, 58.3) == pytest.approx(10.12, abs=0.1)
    assert _calc_diff_pct("N/A", 58.3) == 0.0
    assert _calc_diff_pct(64.2, 0) == 0.0


def test_unit_calc_diff():
    """_calc_diff: 실제 차이값 계산"""
    from structverify.explanation.explainer import _calc_diff
    assert _calc_diff(64.2, 58.3) == "+5.9"
    assert _calc_diff(58.3, 64.2) == "-5.9"
    assert _calc_diff("N/A", 58.3) == "N/A"


def test_unit_format_stat_source_full():
    """_format_stat_source: Evidence에서 출처 텍스트 생성"""
    from structverify.explanation.explainer import _format_stat_source
    from structverify.core.schemas import Evidence
    ev = Evidence(
        source_name="KOSIS 농림어업총조사",
        stat_table_id="DT_1EA1019",
        time_period="2023",
    )
    result = _format_stat_source(ev)
    assert "KOSIS 농림어업총조사" in result
    assert "DT_1EA1019" in result
    assert "2023" in result


def test_unit_format_stat_source_none():
    """_format_stat_source: Evidence 없으면 N/A"""
    from structverify.explanation.explainer import _format_stat_source
    assert _format_stat_source(None) == "N/A"


def test_unit_format_search_hint():
    """_format_search_hint: 독자용 검색 힌트 생성"""
    from structverify.explanation.explainer import _format_search_hint
    from structverify.core.schemas import Claim, ClaimSchema
    claim = Claim(
        doc_id=uuid4(), block_id="b0", sent_id="s0",
        claim_text="65세 이상 과수 농가 64.2%",
        schema=ClaimSchema(
            indicator="고령화비율", population="과수 농가", time_period="2023"
        ),
    )
    hint = _format_search_hint(claim)
    assert "고령화비율" in hint
    assert "2023" in hint


def test_unit_fallback_match(sample_claim, match_result):
    """_fallback_explanation: MATCH fallback 텍스트"""
    from structverify.explanation.explainer import _fallback_explanation
    text = _fallback_explanation(sample_claim, match_result)
    assert "일치" in text
    assert len(text) > 10


def test_unit_fallback_mismatch(sample_claim, mismatch_result):
    """_fallback_explanation: MISMATCH fallback — 수치 포함"""
    from structverify.explanation.explainer import _fallback_explanation
    text = _fallback_explanation(sample_claim, mismatch_result)
    assert "불일치" in text
    assert "64.2" in text or "58.3" in text


def test_unit_fallback_unverifiable(sample_claim, unverifiable_result):
    """_fallback_explanation: UNVERIFIABLE fallback"""
    from structverify.explanation.explainer import _fallback_explanation
    text = _fallback_explanation(sample_claim, unverifiable_result)
    assert "검증 불가" in text


def test_unit_build_prompt_match(sample_claim, match_result):
    """_build_prompt: MATCH 프롬프트에 핵심 필드 포함 확인"""
    from structverify.explanation.explainer import _build_prompt
    prompt = _build_prompt(sample_claim, match_result, "출처: KOSIS")
    assert "64.2" in prompt
    assert "64.1" in prompt
    assert "KOSIS" in prompt
    assert "일치" in prompt.upper() or "MATCH" in prompt.upper()


def test_unit_build_prompt_mismatch(sample_claim, mismatch_result):
    """_build_prompt: MISMATCH 프롬프트에 차이 수치 포함 확인"""
    from structverify.explanation.explainer import _build_prompt
    prompt = _build_prompt(sample_claim, mismatch_result, "출처: KOSIS")
    assert "58.3" in prompt
    assert "64.2" in prompt
    assert "수치 오류" in prompt or "불일치" in prompt


def test_unit_build_prompt_unverifiable(sample_claim, unverifiable_result):
    """_build_prompt: UNVERIFIABLE 프롬프트 생성 확인"""
    from structverify.explanation.explainer import _build_prompt
    prompt = _build_prompt(sample_claim, unverifiable_result, "")
    assert "검증 불가" in prompt or "KOSIS" in prompt


# ── LLM 테스트 (API 키 필요) ───────────────────────────────────────────

@skip_no_llm
async def test_llm_explain_match(sample_claim, match_result, llm_config):
    """LLM — MATCH 설명 생성"""
    from structverify.explanation.explainer import generate_explanation
    explanation = await generate_explanation(
        sample_claim, match_result, {"llm": llm_config}
    )
    print(f"\n[MATCH 설명]\n{explanation}")
    assert isinstance(explanation, str)
    assert len(explanation) > 20
    # provenance_summary도 세팅됐는지
    assert match_result.provenance_summary is not None


@skip_no_llm
async def test_llm_explain_mismatch(sample_claim, mismatch_result, llm_config):
    """LLM — MISMATCH 설명 생성 (차이 원인 포함되어야 함)"""
    from structverify.explanation.explainer import generate_explanation
    explanation = await generate_explanation(
        sample_claim, mismatch_result, {"llm": llm_config}
    )
    print(f"\n[MISMATCH 설명]\n{explanation}")
    assert isinstance(explanation, str)
    assert len(explanation) > 20


@skip_no_llm
async def test_llm_explain_unverifiable(sample_claim, unverifiable_result, llm_config):
    """LLM — UNVERIFIABLE 설명 생성"""
    from structverify.explanation.explainer import generate_explanation
    explanation = await generate_explanation(
        sample_claim, unverifiable_result, {"llm": llm_config}
    )
    print(f"\n[UNVERIFIABLE 설명]\n{explanation}")
    assert isinstance(explanation, str)
    assert len(explanation) > 20


@skip_no_llm
async def test_llm_explain_all_verdicts(sample_claim, match_result, mismatch_result,
                                        unverifiable_result, llm_config):
    """3가지 verdict 설명을 연속으로 생성해서 일관성 확인"""
    from structverify.explanation.explainer import generate_explanation
    config = {"llm": llm_config}

    match_exp = await generate_explanation(sample_claim, match_result, config)
    mismatch_exp = await generate_explanation(sample_claim, mismatch_result, config)
    unverifiable_exp = await generate_explanation(sample_claim, unverifiable_result, config)

    print(f"\n[MATCH]\n{match_exp}")
    print(f"\n[MISMATCH]\n{mismatch_exp}")
    print(f"\n[UNVERIFIABLE]\n{unverifiable_exp}")

    # 3개 설명이 모두 다른 내용이어야 함
    assert match_exp != mismatch_exp
    assert mismatch_exp != unverifiable_exp