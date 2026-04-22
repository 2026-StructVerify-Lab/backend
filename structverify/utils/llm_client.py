"""
utils/llm_client.py — LLM API 통합 클라이언트 (HCX / OpenAI)

Runtime Agent와 Builder Agent가 공유하는 LLM 호출 인터페이스.
모든 LLM 호출은 반드시 이 클래스를 거쳐야 한다 (중앙 tracing).

[김예슬]
- HCX(NCP CLOVA Studio) 및 OpenAI API 실제 호출 구현 담당
- Langfuse 트레이싱 연동 담당
- 모델 분기 (heavy: HCX-003, light: HCX-DASH-001, reasoning: HCX-RAG) 담당

[참고] ReAct (Yao et al., ICLR 2023) — https://github.com/ysymyth/ReAct
  Agent가 Thought → Action(Tool Call) → Observation 순환하는 패턴
"""
from __future__ import annotations
import json, os
from typing import Any
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.provider = self.config.get("provider", "openai")
        self.model = self.config.get("model", "gpt-4o")
        self.temperature = self.config.get("temperature", 0.1)
        self.max_tokens = self.config.get("max_tokens", 2048)
        api_key_env = self.config.get("api_key_env", "LLM_API_KEY")
        self.api_key = os.environ.get(api_key_env, "")

        # TODO [김예슬]: Langfuse 트레이싱 초기화
        #   from langfuse import Langfuse
        #   self.langfuse = Langfuse(
        #       public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        #       secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        #   ) if self.config.get("langfuse_enabled") else None

        # TODO [김예슬]: HCX 전용 클라이언트 초기화
        #   NCP CLOVA Studio API 기본 URL, 인증 헤더 설정

    async def generate(self, prompt: str, system_prompt: str | None = None,
                       temperature: float | None = None) -> str:
        """LLM 프롬프트 전송 → 응답 텍스트 반환"""
        if self.provider == "openai":
            return await self._call_openai(prompt, system_prompt, temperature)
        elif self.provider == "hcx":
            return await self._call_hcx(prompt, system_prompt, temperature)
        raise ValueError(f"미지원 provider: {self.provider}")

    async def generate_json(self, prompt: str, system_prompt: str | None = None) -> dict[str, Any]:
        """LLM 호출 후 JSON 파싱된 응답 반환"""
        raw = await self.generate(prompt, system_prompt)
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            logger.warning(f"JSON 파싱 실패: {raw[:200]}")
            return {"raw": raw}

    async def _call_openai(self, prompt, system_prompt, temperature) -> str:
        """
        OpenAI API 호출

        TODO [김예슬]: openai AsyncClient 실제 호출 구현
          from openai import AsyncOpenAI
          client = AsyncOpenAI(api_key=self.api_key)
          messages = []
          if system_prompt:
              messages.append({"role": "system", "content": system_prompt})
          messages.append({"role": "user", "content": prompt})
          resp = await client.chat.completions.create(
              model=self.model,
              messages=messages,
              temperature=temperature or self.temperature,
              max_tokens=self.max_tokens,
          )
          return resp.choices[0].message.content

        TODO [김예슬]: Langfuse span 기록
          trace = self.langfuse.trace(name="llm_call")
          span = trace.span(name="openai_generate", input={"prompt": prompt})
          ... (호출 후) ...
          span.end(output={"response": result})
        """
        logger.warning("OpenAI stub 호출")
        preview = prompt[:80].replace('"', "'")
        return '{"stub": true, "prompt_preview": "' + preview + '"}'

    async def _call_hcx(self, prompt, system_prompt, temperature) -> str:
        """
        NCP HyperCLOVA X API 호출

        TODO [김예슬]: NCP CLOVA Studio API 규격 구현
          - HCX-003 (중량): claim detection, schema induction, explanation
          - HCX-DASH-001 (경량): domain classification, candidate scoring
          - HCX-RAG (추론): 복합 근거 종합 판단

          import httpx
          headers = {
              "X-NCP-CLOVASTUDIO-API-KEY": self.api_key,
              "X-NCP-APIGW-API-KEY": os.environ.get("NCP_APIGW_API_KEY", ""),
              "Content-Type": "application/json",
          }
          payload = {
              "messages": [
                  {"role": "system", "content": system_prompt or ""},
                  {"role": "user", "content": prompt},
              ],
              "maxTokens": self.max_tokens,
              "temperature": temperature or self.temperature,
          }
          model_id = self.config.get("models", {}).get("heavy", "HCX-003")
          url = f"https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/{model_id}"
          async with httpx.AsyncClient() as client:
              resp = await client.post(url, json=payload, headers=headers, timeout=30)
              resp.raise_for_status()
              data = resp.json()
              return data["result"]["message"]["content"]

        TODO [김예슬]: 모델 선택 로직
          - 태스크별 heavy/light 모델 자동 분기
          - config.llm.models.heavy / light / reasoning 사용
        """
        logger.warning("HCX stub 호출")
        preview = prompt[:80].replace('"', "'")
        return '{"stub": true, "prompt_preview": "' + preview + '"}'
