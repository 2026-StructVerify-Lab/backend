"""
LLMClient 동작 확인용 빠른 테스트

실행 방법:
    # 환경변수 세팅
    export NCP_API_KEY="your-ncp-api-key-here"

    # 전체 테스트
    cd backend
    python -m pytest tests/test_llm_client.py -v -s

    # 특정 테스트만
    python -m pytest tests/test_llm_client.py::test_hcx_generate -v -s
"""
import asyncio
import os
import pytest

# API 키 없으면 스킵
pytestmark = pytest.mark.skipif(
    not os.environ.get("NCP_API_KEY"),
    reason="NCP_API_KEY 환경변수 없음 — export NCP_API_KEY=xxx 후 실행"
)


@pytest.fixture
def hcx_config():
    return {
        "provider": "hcx",
        "models": {
            "heavy": "HCX-003",
            "light": "HCX-DASH-001",
        },
        "temperature": 0.1,
        "max_tokens": 512,
        "api_key_env": "NCP_API_KEY",
    }


# ── 1. 기본 generate 테스트 ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hcx_generate(hcx_config):
    """HCX 텍스트 생성 기본 동작 확인"""
    from structverify.utils.llm_client import LLMClient
    client = LLMClient(config=hcx_config)

    result = await client.generate(
        prompt="안녕하세요. 한 문장으로 자기소개해주세요.",
        system_prompt="당신은 친절한 AI입니다.",
    )
    print(f"\n[generate 결과]\n{result}")
    assert isinstance(result, str)
    assert len(result) > 0


# ── 2. generate_json 테스트 ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hcx_generate_json(hcx_config):
    """JSON 응답 파싱 확인 — candidate scorer 프롬프트 형식으로 테스트"""
    from structverify.utils.llm_client import LLMClient
    client = LLMClient(config=hcx_config)

    prompt = """아래 문장이 공식 통계로 검증 가능한 수치 기반 주장인지 판단하세요.

문장: "2023년 기준 국내 65세 이상 고령 인구 비율은 18.4%를 넘어섰다."

JSON으로만 답하세요:
{"candidate_score": 0.0, "candidate_label": false, "reason": "짧은 근거"}"""

    result = await client.generate_json(
        prompt=prompt,
        system_prompt="팩트체크 candidate detector. JSON으로만 답하세요.",
        model_tier="light",  # HCX-DASH-001 경량 모델 사용
    )
    print(f"\n[generate_json 결과]\n{result}")
    assert "candidate_score" in result or "raw" in result


# ── 3. 경량 모델(HCX-DASH) 테스트 ────────────────────────────────────────

@pytest.mark.asyncio
async def test_hcx_light_model(hcx_config):
    """경량 모델(HCX-DASH-001) 응답 속도 및 품질 확인"""
    from structverify.utils.llm_client import LLMClient
    import time

    client = LLMClient(config=hcx_config)

    start = time.time()
    result = await client.generate_light(
        prompt="다음 중 농업 관련 도메인은? agriculture / economy / healthcare",
        system_prompt="도메인 분류기. 단답으로만 답하세요.",
    )
    elapsed = time.time() - start

    print(f"\n[경량 모델 결과] ({elapsed:.2f}초)\n{result}")
    assert isinstance(result, str)
    assert len(result) > 0


# ── 4. 에러 핸들링 테스트 ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_invalid_api_key():
    """잘못된 API 키 에러 핸들링 확인"""
    from structverify.utils.llm_client import LLMClient
    import httpx

    client = LLMClient(config={
        "provider": "hcx",
        "models": {"heavy": "HCX-003"},
        "api_key_env": "NONEXISTENT_KEY_FOR_TEST",
    })
    # API 키가 없으므로 인증 오류 발생해야 함
    with pytest.raises((httpx.HTTPStatusError, RuntimeError, Exception)):
        await client.generate("테스트")


# ── 5. JSON 파싱 유틸 단위 테스트 ─────────────────────────────────────────

def test_parse_json_response_clean():
    """깔끔한 JSON 파싱"""
    from structverify.utils.llm_client import _parse_json_response
    raw = '{"score": 0.8, "label": true}'
    result = _parse_json_response(raw)
    assert result["score"] == 0.8
    assert result["label"] is True


def test_parse_json_response_with_codeblock():
    """코드블록 포함 JSON 파싱"""
    from structverify.utils.llm_client import _parse_json_response
    raw = '```json\n{"score": 0.9, "label": true}\n```'
    result = _parse_json_response(raw)
    assert result["score"] == 0.9


def test_parse_json_response_fallback():
    """파싱 실패 시 raw 반환"""
    from structverify.utils.llm_client import _parse_json_response
    raw = "죄송합니다, JSON 형식으로 답변드리겠습니다."
    result = _parse_json_response(raw)
    assert "raw" in result