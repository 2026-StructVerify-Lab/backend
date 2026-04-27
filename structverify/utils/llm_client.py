"""
utils/llm_client.py — LLM API 통합 클라이언트 (HCX / OpenAI)

Runtime Agent와 Builder Agent가 공유하는 LLM 호출 인터페이스.
모든 LLM 호출은 반드시 이 클래스를 거쳐야 한다 (중앙 tracing).

[김예슬 - 2026-04-22]
- _call_hcx(): NCP CLOVA Studio Chat Completions v1 API 실제 호출 구현
- _call_openai(): AsyncOpenAI 클라이언트 실제 호출 구현 (HCX 대체용)
- model_tier 파라미터 추가: "heavy"(HCX-003) / "light"(HCX-DASH-001) / "reasoning" 자동 분기
- generate_light() / generate_json_light() 단축 메서드 추가
- _parse_json_response(): 클래스 외부 모듈 함수로 분리 (단위 테스트 용이)
- _direct_api_key 지원 (테스트용 키 직접 주입)

[김예슬 - 2026-04-24]
- CLOVA Studio v3 API 지원 추가
  · _call_hcx_v3(): Chat Completions v3 (HCX-005, HCX-DASH-002)
  · _call_hcx_structured(): Structured Outputs (HCX-007 전용)
    - responseFormat.type = "json" + JSON Schema 정의
    - JSON 파싱 실패 없음 — 항상 스키마 형식으로 반환 보장
- generate_structured(): Structured Outputs 전용 공개 메서드 추가
- 모델 tier 업데이트:
  · heavy: HCX-003 → 복잡한 태스크 (check-worthiness, 설명 생성)
  · light: HCX-DASH-002 → 빠른 분류 (도메인, candidate scoring)
  · structured: HCX-007 → JSON 구조화 추출 (schema_inductor 전용)

[참고] CLOVA Studio Structured Outputs
  https://api.ncloud-docs.com/docs/en/clovastudio-chatcompletionsv3-so
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Any

import httpx

from structverify.utils.logger import get_logger

logger = get_logger(__name__)

# ── 엔드포인트 ────────────────────────────────────────────────────────────
HCX_V1_BASE = "https://clovastudio.stream.ntruss.com/v1/chat-completions"
HCX_V3_BASE = "https://clovastudio.stream.ntruss.com/v3/chat-completions"


class LLMClient:
    """
    HCX(NCP CLOVA Studio) + OpenAI 통합 클라이언트.

    config 예시 (default.yaml의 llm 섹션):
        provider: "hcx"
        models:
            heavy:      "HCX-003"       # 복잡한 태스크 (v1 API)
            light:      "HCX-DASH-002"  # 빠른 분류 (v3 API)
            structured: "HCX-007"       # JSON 구조화 (v3 Structured Outputs)
        temperature: 0.1
        max_tokens: 2048
        api_key_env: "CLOVASTUDIO_API_KEY"
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.provider = self.config.get("provider", "hcx")
        self.models = self.config.get("models", {
            "heavy":      "HCX-003",
            "light":      "HCX-DASH-002",
            "structured": "HCX-007",
            "reasoning":  "HCX-003",
        })
        self.default_model = self.models.get("heavy", "HCX-003")
        self.temperature = float(self.config.get("temperature", 0.1))
        self.max_tokens = int(self.config.get("max_tokens", 2048))

        # HCX API 키 — _direct_api_key로 테스트 시 직접 주입 가능
        api_key_env = self.config.get("api_key_env", "NCP_API_KEY")
        if self.config.get("_direct_api_key"):
            self.api_key = self.config["_direct_api_key"]
        else:
            self.api_key = os.environ.get(api_key_env, "")
        if not self.api_key:
            logger.warning(f"LLM API 키 없음 — 환경변수 {api_key_env} 확인 필요")

        # OpenAI API 키
        openai_key_env = self.config.get("openai_key_env", "OPENAI_API_KEY")
        if openai_key_env.startswith("sk-"):
            self.openai_api_key = openai_key_env
        else:
            self.openai_api_key = os.environ.get(openai_key_env, "")

    # ── 공개 인터페이스 ──────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        model_tier: str = "heavy",
    ) -> str:
        """텍스트 생성 — 일반 Chat Completions"""
        if self.provider == "hcx":
            model = self.models.get(model_tier, self.default_model)
            # HCX-DASH-002, HCX-005 → v3 API
            if model in ("HCX-DASH-002", "HCX-005"):
                return await self._call_hcx_v3(prompt, system_prompt, temperature, model)
            # HCX-003 → v1 API
            return await self._call_hcx_v1(prompt, system_prompt, temperature, model)
        elif self.provider == "openai":
            return await self._call_openai(prompt, system_prompt, temperature, model_tier)
        raise ValueError(f"미지원 provider: {self.provider}")

    async def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_tier: str = "heavy",
    ) -> dict[str, Any]:
        """텍스트 생성 후 JSON 파싱 — 파싱 실패 시 {"raw": ...} 반환"""
        raw = await self.generate(prompt, system_prompt, model_tier=model_tier)
        return _parse_json_response(raw)

    async def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Structured Outputs — HCX-007 전용.
        JSON Schema를 정의하면 항상 그 형식으로 반환 보장.
        파싱 실패 없음.

        Args:
            prompt: 사용자 프롬프트
            schema: JSON Schema 정의 (object 타입)
            system_prompt: 시스템 프롬프트

        Returns:
            스키마에 맞는 dict

        사용 예:
            result = await llm.generate_structured(
                prompt="주장: '2023년 고령화율 64.2%'에서 정보 추출",
                schema={
                    "type": "object",
                    "properties": {
                        "indicator": {"type": "string"},
                        "value": {"type": "number"},
                    },
                    "required": ["indicator", "value"]
                }
            )
        """
        if self.provider == "hcx":
            return await self._call_hcx_structured(prompt, schema, system_prompt)
        elif self.provider == "openai":
            # OpenAI도 response_format으로 JSON Schema 지원
            raw = await self._call_openai_structured(prompt, schema, system_prompt)
            return raw
        raise ValueError(f"미지원 provider: {self.provider}")

    async def generate_light(self, prompt: str, system_prompt: str | None = None) -> str:
        """경량 모델(HCX-DASH-002) 호출 단축키"""
        return await self.generate(prompt, system_prompt, model_tier="light")

    async def generate_json_light(self, prompt: str, system_prompt: str | None = None) -> dict[str, Any]:
        """경량 모델로 JSON 응답 생성"""
        raw = await self.generate_light(prompt, system_prompt)
        return _parse_json_response(raw)

    # ── HCX v1 (HCX-003) ────────────────────────────────────────────────────

    async def _call_hcx_v1(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float | None,
        model: str,
    ) -> str:
        """
        NCP CLOVA Studio Chat Completions v1 API.
        엔드포인트: POST /v1/chat-completions/{model}
        대상 모델: HCX-003
        """
        url = f"{HCX_V1_BASE}/{model}"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "maxTokens": self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "topP": 0.8,
            "topK": 0,
            "repeatPenalty": 5.0,
            "stopBefore": [],
            "includeAiFilters": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            status_code = data.get("status", {}).get("code", "")
            if status_code != "20000":
                msg = data.get("status", {}).get("message", "unknown")
                raise RuntimeError(f"HCX v1 API 오류: {status_code} {msg}")
            content = data["result"]["message"]["content"]
            logger.debug(f"HCX v1 응답 ({model}): {content[:80]}...")
            return content
        except httpx.HTTPStatusError as e:
            logger.error(f"HCX v1 HTTP 오류: {e.response.status_code} — {e.response.text[:200]}")
            raise
        except httpx.TimeoutException:
            logger.error(f"HCX v1 타임아웃: {url}")
            raise

    # ── HCX v3 (HCX-005, HCX-DASH-002) ─────────────────────────────────────

    async def _call_hcx_v3(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float | None,
        model: str,
    ) -> str:
        """
        NCP CLOVA Studio Chat Completions v3 API.
        엔드포인트: POST /v3/chat-completions/{model}
        대상 모델: HCX-005 (비전), HCX-DASH-002 (경량)
        """
        url = f"{HCX_V3_BASE}/{model}"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "maxCompletionTokens": self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "topP": 0.8,
            "topK": 0,
            "repetitionPenalty": 1.1,
            "stop": [],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            status_code = data.get("status", {}).get("code", "")
            if status_code != "20000":
                msg = data.get("status", {}).get("message", "unknown")
                raise RuntimeError(f"HCX v3 API 오류: {status_code} {msg}")
            content = data["result"]["message"]["content"]
            logger.debug(f"HCX v3 응답 ({model}): {content[:80]}...")
            return content
        except httpx.HTTPStatusError as e:
            logger.error(f"HCX v3 HTTP 오류: {e.response.status_code} — {e.response.text[:200]}")
            raise

    # ── HCX Structured Outputs (HCX-007 전용) ───────────────────────────────

    async def _call_hcx_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_prompt: str | None,
    ) -> dict[str, Any]:
        """
        NCP CLOVA Studio Structured Outputs API.
        엔드포인트: POST /v3/chat-completions/HCX-007
        반환: JSON Schema에 맞는 dict (파싱 실패 없음)

        주의:
        - HCX-007 모델 전용
        - thinking과 동시 사용 불가 → thinking.effort: "none" 고정
        - responseFormat.type = "json" + schema 정의 필수
        """
        url = f"{HCX_V3_BASE}/HCX-007"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "maxCompletionTokens": self.max_tokens,
            "temperature": self.temperature,
            "topP": 0.8,
            "topK": 0,
            "repetitionPenalty": 1.1,
            "stop": [],
            "thinking": {"effort": "none"},  # Structured Outputs와 thinking 동시 불가
            "responseFormat": {
                "type": "json",
                "schema": schema,
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            status_code = data.get("status", {}).get("code", "")
            if status_code != "20000":
                msg = data.get("status", {}).get("message", "unknown")
                raise RuntimeError(f"HCX Structured Outputs 오류: {status_code} {msg}")

            content = data["result"]["message"]["content"]
            logger.debug(f"HCX Structured 응답: {content[:80]}...")

            # content가 이미 JSON Schema 형식이므로 바로 파싱
            return json.loads(content)

        except httpx.HTTPStatusError as e:
            logger.error(f"HCX Structured HTTP 오류: {e.response.status_code} — {e.response.text[:200]}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"HCX Structured JSON 파싱 실패 (스키마 오류?): {e}")
            raise

    # ── OpenAI ───────────────────────────────────────────────────────────────

    async def _call_openai(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float | None,
        model_tier: str,
    ) -> str:
        """OpenAI Chat Completions API 호출"""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("pip install openai")

        model_map = {"heavy": "gpt-4o", "light": "gpt-4o-mini",
                     "structured": "gpt-4o", "reasoning": "gpt-4o"}
        model = model_map.get(model_tier, "gpt-4o")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        client = AsyncOpenAI(api_key=self.openai_api_key)
        resp = await client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content

    async def _call_openai_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_prompt: str | None,
    ) -> dict[str, Any]:
        """OpenAI JSON Schema 구조화 응답"""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("pip install openai")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        client = AsyncOpenAI(api_key=self.openai_api_key)
        resp = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_schema", "json_schema": {
                "name": "structured_output", "strict": True, "schema": schema
            }},
        )
        content = resp.choices[0].message.content
        return json.loads(content)


# ── 유틸리티 ────────────────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> dict[str, Any]:
    """
    LLM 응답에서 JSON 추출 + 파싱.
    ```json 코드블록 / 일반 코드블록 / 순수 JSON 3가지 케이스 처리.
    실패 시 {"raw": 원문} 반환.
    """
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"JSON 파싱 실패: {text[:200]}")
        return {"raw": raw}