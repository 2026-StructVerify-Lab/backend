"""
utils/llm_client.py — LLM API 통합 클라이언트 (HCX / OpenAI)

Runtime Agent와 Builder Agent가 공유하는 LLM 호출 인터페이스.
모든 LLM 호출은 반드시 이 클래스를 거쳐야 한다 (중앙 tracing).

[김예슬]
- HCX(NCP CLOVA Studio) 및 OpenAI API 실제 호출 구현 담당
- Langfuse 트레이싱 연동 담당
- 태스크별 heavy/light 모델 자동 분기 담당

[김예슬 - 2026-04-22]
- _call_hcx(): NCP CLOVA Studio Chat Completions API 실제 호출 구현
  · 엔드포인트: https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/{model}
  · 인증 헤더: Authorization Bearer + X-NCP-CLOVASTUDIO-REQUEST-ID
  · Accept: application/json 으로 비스트리밍 JSON 응답 처리
  · 응답 파싱: data["result"]["message"]["content"] 추출
  · httpx.AsyncClient 기반 비동기 호출 + 타임아웃/HTTP 에러 처리
- _call_openai(): AsyncOpenAI 클라이언트 실제 호출 구현 (HCX 대체용)
- model_tier 파라미터 추가: "heavy"(HCX-003) / "light"(HCX-DASH-001) / "reasoning" 자동 분기
- generate_light() / generate_json_light() 단축 메서드 추가
- _parse_json_response(): 클래스 외부 모듈 함수로 분리 (단위 테스트 용이)
  · ```json 코드블록 / 일반 코드블록 / 순수 JSON 3가지 케이스 처리
- TODO: Langfuse 트레이싱 초기화 구조 주석으로 작성 (미구현)

[참고] ReAct (Yao et al., ICLR 2023) — https://github.com/ysymyth/ReAct
  Agent가 Thought → Action(Tool Call) → Observation 순환하는 패턴
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Any

import httpx

from structverify.utils.logger import get_logger

logger = get_logger(__name__)

# ── HCX 엔드포인트 ─────────────────────────────────────────────────────────
# docs: https://api.ncloud-docs.com/docs/en/clovastudio-chatcompletions
# Accept: application/json → 일반 JSON 응답 (스트리밍 X)
# Accept: text/event-stream → SSE 스트리밍 응답
HCX_BASE_URL = "https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions"


class LLMClient:
    """
    HCX(NCP CLOVA Studio) + OpenAI 통합 클라이언트.

    config 예시 (default.yaml의 llm 섹션):
        provider: "hcx"
        models:
            heavy: "HCX-003"       # claim detection, schema induction, explanation
            light: "HCX-DASH-001"  # domain classification, candidate scoring
            reasoning: "HCX-003"   # complex reasoning
        temperature: 0.1
        max_tokens: 2048
        api_key_env: "NCP_API_KEY"
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.provider = self.config.get("provider", "hcx")
        self.models = self.config.get("models", {
            "heavy": "HCX-003",
            "light": "HCX-DASH-001",
            "reasoning": "HCX-003",
        })
        # 기본 모델은 heavy (중량 모델)
        self.default_model = self.models.get("heavy", "HCX-003")
        self.temperature = float(self.config.get("temperature", 0.1))
        self.max_tokens = int(self.config.get("max_tokens", 2048))

        # NCP API 키 — env에서 읽음
        api_key_env = self.config.get("api_key_env", "NCP_API_KEY")
        self.api_key = os.environ.get(api_key_env, "")
        if not self.api_key:
            logger.warning(f"LLM API 키 없음 — 환경변수 {api_key_env} 확인 필요")

        # OpenAI API 키
        openai_key_env = self.config.get("openai_key_env", "OPENAI_API_KEY")
        self.openai_api_key = os.environ.get(openai_key_env, "")

        # TODO [김예슬]: Langfuse 트레이싱 초기화
        # from langfuse import Langfuse
        # self.langfuse = Langfuse(
        #     public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        #     secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        #     host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        # ) if self.config.get("langfuse_enabled") else None
        self.langfuse = None

    # ── 공개 인터페이스 ──────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        model_tier: str = "heavy",  # "heavy" | "light" | "reasoning"
    ) -> str:
        """
        LLM 프롬프트 전송 → 응답 텍스트 반환.

        Args:
            prompt: 사용자 입력 프롬프트
            system_prompt: 시스템 프롬프트 (없으면 기본값 사용)
            temperature: None이면 config 기본값 사용
            model_tier: "heavy"(HCX-003), "light"(HCX-DASH-001), "reasoning"(HCX-003)
        """
        if self.provider == "hcx":
            return await self._call_hcx(prompt, system_prompt, temperature, model_tier)
        elif self.provider == "openai":
            return await self._call_openai(prompt, system_prompt, temperature, model_tier)
        raise ValueError(f"미지원 provider: {self.provider}")

    async def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_tier: str = "heavy",
    ) -> dict[str, Any]:
        """
        LLM 호출 후 JSON 파싱된 응답 반환.
        JSON 파싱 실패 시 {"raw": 원문} 반환.
        """
        raw = await self.generate(prompt, system_prompt, model_tier=model_tier)
        return _parse_json_response(raw)

    async def generate_light(self, prompt: str, system_prompt: str | None = None) -> str:
        """경량 모델(HCX-DASH-001) 호출 — 도메인 분류, candidate scoring 등."""
        return await self.generate(prompt, system_prompt, model_tier="light")

    async def generate_json_light(self, prompt: str, system_prompt: str | None = None) -> dict[str, Any]:
        """경량 모델로 JSON 응답 생성."""
        raw = await self.generate_light(prompt, system_prompt)
        return _parse_json_response(raw)

    # ── HCX 구현 ────────────────────────────────────────────────────────────

    async def _call_hcx(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float | None,
        model_tier: str = "heavy",
    ) -> str:
        """
        NCP CLOVA Studio Chat Completions API 호출.

        엔드포인트: POST https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/{model}
        인증: Authorization: Bearer {API_KEY}
        응답: Accept: application/json → JSON 응답 (비스트리밍)

        응답 구조:
            {
              "status": {"code": "20000", "message": "OK"},
              "result": {
                "message": {"role": "assistant", "content": "..."},
                "stopReason": "end_token",
                "inputLength": 100,
                "outputLength": 50
              }
            }
        """
        model = self.models.get(model_tier, self.default_model)
        url = f"{HCX_BASE_URL}/{model}"

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
            "Accept": "application/json",  # 비스트리밍 JSON 응답
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            # 응답 파싱
            status_code = data.get("status", {}).get("code", "")
            if status_code != "20000":
                msg = data.get("status", {}).get("message", "unknown error")
                logger.error(f"HCX API 오류: {status_code} — {msg}")
                raise RuntimeError(f"HCX API error: {status_code} {msg}")

            content = data["result"]["message"]["content"]
            logger.debug(f"HCX 응답 ({model}): {content[:100]}...")
            return content

        except httpx.HTTPStatusError as e:
            logger.error(f"HCX HTTP 오류: {e.response.status_code} — {e.response.text[:200]}")
            raise
        except httpx.TimeoutException:
            logger.error(f"HCX 타임아웃: {url}")
            raise
        except KeyError as e:
            logger.error(f"HCX 응답 파싱 실패 — 필드 없음: {e}\n응답: {data}")
            raise

    # ── OpenAI 구현 ──────────────────────────────────────────────────────────

    async def _call_openai(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float | None,
        model_tier: str = "heavy",
    ) -> str:
        """
        OpenAI Chat Completions API 호출.
        HCX 없을 때 대체 또는 실험용.
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai 패키지 필요: pip install openai")

        # OpenAI 모델 매핑 (HCX 모델명 → OpenAI 모델명)
        openai_model_map = {
            "heavy": "gpt-4o",
            "light": "gpt-4o-mini",
            "reasoning": "gpt-4o",
        }
        model = openai_model_map.get(model_tier, "gpt-4o")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        client = AsyncOpenAI(api_key=self.openai_api_key)
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=self.max_tokens,
        )
        content = resp.choices[0].message.content
        logger.debug(f"OpenAI 응답 ({model}): {content[:100]}...")
        return content


# ── 유틸리티 ────────────────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> dict[str, Any]:
    """
    LLM 응답 텍스트에서 JSON 추출 + 파싱.

    처리 순서:
    1) ```json ... ``` 코드블록 추출
    2) ``` ... ``` 일반 코드블록 추출
    3) 그대로 파싱 시도
    4) 실패 시 {"raw": 원문} 반환
    """
    text = raw.strip()

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"JSON 파싱 실패 (앞 200자): {text[:200]}")
        return {"raw": raw}