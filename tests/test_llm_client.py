"""
tests/test_llm_client.py — LLMClient 동작 확인용 빠른 테스트

실행 방법:
    # 1) 패키지 설치 (처음 한 번만)
    cd backend
    pip install -e ".[dev]"      # pytest-asyncio 포함 dev 의존성 전체 설치
    pip install httpx             # llm_client 의존성

    # 2) API 키 없이도 되는 테스트 (JSON 파싱 단위 테스트)
    python -m pytest tests/test_llm_client.py -k "parse" -v

    # 3) API 키 필요한 테스트
    export NCP_API_KEY="발급받은키"
    python -m pytest tests/test_llm_client.py -v -s
"""
import os
import pytest

# API 키 없으면 비동기 테스트 전체 스킵
HAS_API_KEY = bool(os.environ.get("NCP_API_KEY"))
skip_without_key = pytest.mark.skipif(
    not HAS_API_KEY,
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

@skip_without_key
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

@skip_without_key
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
        model_tier="light",
    )
    print(f"\n[generate_json 결과]\n{result}")
    assert "candidate_score" in result or "raw" in result


# ── 3. 경량 모델(HCX-DASH) 테스트 ────────────────────────────────────────

@skip_without_key
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

async def test_invalid_api_key():
    """잘못된 API 키 → 예외 발생 확인"""
    from structverify.utils.llm_client import LLMClient
    import httpx

    client = LLMClient(config={
        "provider": "hcx",
        "models": {"heavy": "HCX-003"},
        "api_key_env": "NONEXISTENT_KEY_FOR_TEST",  # 의도적으로 없는 키
    })
    with pytest.raises((httpx.HTTPStatusError, RuntimeError, Exception)):
        await client.generate("테스트")


# ── 5. JSON 파싱 단위 테스트 (API 키 불필요) ─────────────────────────────

def test_parse_json_clean():
    """깔끔한 JSON 파싱"""
    from structverify.utils.llm_client import _parse_json_response
    result = _parse_json_response('{"score": 0.8, "label": true}')
    assert result["score"] == 0.8
    assert result["label"] is True


def test_parse_json_with_codeblock():
    """코드블록 포함 JSON 파싱"""
    from structverify.utils.llm_client import _parse_json_response
    result = _parse_json_response('```json\n{"score": 0.9, "label": true}\n```')
    assert result["score"] == 0.9


def test_parse_json_plain_codeblock():
    """``` 일반 코드블록 파싱"""
    from structverify.utils.llm_client import _parse_json_response
    result = _parse_json_response('```\n{"score": 0.7}\n```')
    assert result["score"] == 0.7


def test_parse_json_fallback():
    """파싱 실패 시 raw 반환"""
    from structverify.utils.llm_client import _parse_json_response
    result = _parse_json_response("죄송합니다, 답변드리겠습니다.")
    assert "raw" in result


# ── 5. provider별 모델 티어 매핑 (#64) — 키/네트워크 불필요 ─────────────────

@pytest.mark.parametrize("provider,heavy,light", [
    ("hcx",     "HCX-003",       "HCX-DASH-002"),
    ("openai",  "gpt-4o",        "gpt-4o-mini"),
    ("upstage", "solar-pro2",    "solar-mini"),
    ("gemini",  "gemini-2.5-pro", "gemini-2.5-flash"),
])
def test_provider_default_models(provider, heavy, light):
    """models 미지정 시 provider별 기본 모델 사용."""
    from structverify.utils.llm_client import LLMClient
    c = LLMClient({"provider": provider})
    assert c.models["heavy"] == heavy
    assert c.models["light"] == light
    assert c.default_model == heavy


def test_provider_switch_ignores_hcx_models():
    """default.yaml의 HCX models를 둔 채 provider만 바꿔도 provider 기본 모델 사용 (#64 핵심).
    이게 안 되면 upstage에 model='HCX-003'을 보내 404."""
    from structverify.utils.llm_client import LLMClient
    hcx_models = {"heavy": "HCX-003", "light": "HCX-DASH-002", "structured": "HCX-007"}
    c = LLMClient({"provider": "upstage", "models": hcx_models})
    assert c.models["light"] == "solar-mini"
    assert not c.models["structured"].upper().startswith("HCX")


def test_provider_partial_model_override():
    """provider용 이름으로 일부 티어만 override하면 나머지는 기본값 merge."""
    from structverify.utils.llm_client import LLMClient
    c = LLMClient({"provider": "upstage", "models": {"heavy": "solar-pro3"}})
    assert c.models["heavy"] == "solar-pro3"      # override 반영
    assert c.models["light"] == "solar-mini"      # 나머지는 provider 기본값


def test_hcx_user_models_preserved():
    """hcx는 기존대로 사용자 지정 models를 그대로 존중(회귀 방지)."""
    from structverify.utils.llm_client import LLMClient
    c = LLMClient({"provider": "hcx", "models": {"heavy": "HCX-003", "light": "HCX-DASH-001"}})
    assert c.models["light"] == "HCX-DASH-001"


@pytest.mark.asyncio
async def test_openai_structured_uses_compatible_path(monkeypatch):
    """OpenAI structured는 strict json_schema 대신 compatible(json_object+힌트) 경로."""
    from structverify.utils.llm_client import LLMClient

    calls: list[dict] = []

    async def fake_compatible(self, prompt, schema, system_prompt, *, base_url, api_key):
        calls.append({
            "prompt": prompt,
            "schema": schema,
            "system_prompt": system_prompt,
            "base_url": base_url,
            "api_key": api_key,
        })
        return {"schemas": []}

    monkeypatch.setattr(
        "structverify.utils.llm_client.LLMClient._call_openai_compatible_structured",
        fake_compatible,
    )

    client = LLMClient({"provider": "openai", "openai_key_env": "sk-test"})
    out = await client.generate_structured(
        prompt="test",
        schema={"type": "object", "properties": {"schemas": {"type": "array"}}},
        system_prompt="sys",
    )

    assert out == {"schemas": []}
    assert len(calls) == 1
    assert calls[0]["base_url"] == "https://api.openai.com/v1"
    assert calls[0]["api_key"]  # openai_key_env / _direct sk- 경로
