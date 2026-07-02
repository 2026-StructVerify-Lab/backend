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
   

# [박재윤 - 2026-05-14]: _call_hcx_v1, _call_hcx_v3 429 retry backoff 추가
 - 429 rate limit 시 exponential backoff 재시도{V1,V3} (최대 3회)

    NCP CLOVA Studio Chat Completions v1,V3 API.
    엔드포인트: POST /v1/chat-completions/{model}
    대상 모델: HCX-003
    
# [박재윤 - 2026-05-18]: _call_hcx_structured 429 retry backoff 추가
#   · _call_hcx_v1, _call_hcx_v3와 동일한 exponential backoff 적용 (최대 3회)
#   · schema_inductor generate_structured 호출 시 rate limit 문제 해결
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any

import httpx

from structverify.utils.logger import get_logger

logger = get_logger(__name__)

# ── 엔드포인트 ────────────────────────────────────────────────────────────
HCX_V1_BASE = "https://clovastudio.stream.ntruss.com/v1/chat-completions"
HCX_V3_BASE = "https://clovastudio.stream.ntruss.com/v3/chat-completions"


# ── [2026-05-21] 프로세스 전역 HCX rate limiter ───────────────────────────
# HCX는 같은 API 키에 *초당 요청 수* 한도가 있어, claim_detector(concurrency=4) +
# planner + reflect 등 *여러 코드 경로*가 짧은 시간 안에 동시에 호출하면 429 폭주.
# concurrency 제한만으론 phase 간 burst를 못 막으므로, 모든 HCX 호출(v1/v3/structured)
# 직전에 acquire()를 통과시켜 *호출 사이 최소 간격*을 강제한다.
# min_interval 단위: 초. 0이면 비활성.
class _HCXRateLimiter:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._last_call_ts: float = 0.0
        self._min_interval: float = 0.0
        self._headers_logged: bool = False

    def configure(self, interval_sec: float) -> None:
        """max(기존, 새 값)로 갱신 — 더 빡센 설정이 이김."""
        if interval_sec > self._min_interval:
            self._min_interval = interval_sec
            logger.info(
                f"[hcx_rate_limiter] min_interval={interval_sec:.3f}s "
                f"(= ≤{1.0/interval_sec:.1f} req/sec)"
            )

    async def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_ts
            wait = self._min_interval - elapsed
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call_ts = time.monotonic()

    def _log_headers_if_first(self, headers: Any) -> None:
        """첫 429 응답 헤더의 rate-limit 관련 키를 한 번만 로그. HCX 실제 한도 파악용."""
        if self._headers_logged:
            return
        self._headers_logged = True
        try:
            keys_of_interest = [
                k for k in headers.keys()
                if any(t in k.lower() for t in ("ratelimit", "rate-limit", "retry-after", "quota"))
            ]
            if keys_of_interest:
                info = {k: headers.get(k) for k in keys_of_interest}
                logger.warning(f"[hcx_rate_limiter] HCX 429 응답 헤더 (한도 진단용): {info}")
            else:
                # 알려진 key 없으면 *모든 헤더 키* 한 번 로그 (이름 단서라도)
                logger.warning(
                    f"[hcx_rate_limiter] HCX 429 응답 헤더 키 (rate-limit 관련 없음): "
                    f"{list(headers.keys())[:20]}"
                )
        except Exception as _e:
            logger.debug(f"[hcx_rate_limiter] 헤더 로그 실패: {_e}")


_hcx_rate_limiter = _HCXRateLimiter()


# [v2 - 김예슬 / #64] provider별 기본 모델 티어 매핑.
#   config.llm.models 를 안 바꾸고 provider 만 바꿔도 동작하도록, provider별 기본 모델을 둔다.
#   (default.yaml 의 models 는 HCX 이름이라, provider 만 upstage/gemini 로 바꾸면 그대로 쓰면 404)
_PROVIDER_DEFAULT_MODELS = {
    "hcx":     {"heavy": "HCX-003",       "light": "HCX-DASH-002",      "structured": "HCX-007",       "reasoning": "HCX-003"},
    "openai":  {"heavy": "gpt-4o",        "light": "gpt-4o-mini",       "structured": "gpt-4o",        "reasoning": "gpt-4o"},
    "upstage": {"heavy": "solar-pro2",    "light": "solar-mini",        "structured": "solar-pro2",    "reasoning": "solar-pro2"},
    "gemini":  {"heavy": "gemini-2.5-pro", "light": "gemini-2.5-flash", "structured": "gemini-2.5-pro", "reasoning": "gemini-2.5-pro"},
}


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
        # [v2 - 김예슬 / #64] provider별 모델 티어 매핑 — provider 만 바꿔도 동작하게.
        #   · config.models 없으면 provider 기본값.
        #   · provider!=hcx 인데 config.models 가 HCX 이름이면(=default.yaml 잔재) 무시하고 provider 기본값.
        #   · 그 외엔 provider 기본값 위에 config.models 를 덮어써 일부 티어만 override 허용.
        _defaults = _PROVIDER_DEFAULT_MODELS.get(self.provider, _PROVIDER_DEFAULT_MODELS["hcx"])
        _user_models = self.config.get("models")
        if not _user_models:
            self.models = dict(_defaults)
        elif self.provider != "hcx" and any(
            str(v).upper().startswith("HCX") for v in _user_models.values()
        ):
            self.models = dict(_defaults)
        else:
            self.models = {**_defaults, **_user_models}
        self.default_model = self.models.get("heavy") or next(iter(self.models.values()), "HCX-003")
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

        # [v2 - 김예슬] Upstage(Solar) 지원 — OpenAI 호환 API (base_url만 다름).
        #   provider: "upstage" 로 두면 HCX 키 없이 Upstage 키로 동작. (pip install openai 필요)
        self.upstage_base_url = self.config.get("base_url", "https://api.upstage.ai/v1")
        self.upstage_api_key = (
            self.config.get("_direct_api_key")
            or os.environ.get(self.config.get("api_key_env", "UPSTAGE_API_KEY"), "")
            or os.environ.get("UPSTAGE_API_KEY", "")
        )
        # (upstage 모델 티어는 _PROVIDER_DEFAULT_MODELS["upstage"]에서 결정 — #64)

        # [v2 - 김예슬] Gemini(Google) 지원 — OpenAI 호환 엔드포인트.
        #   provider: "gemini" + GEMINI_API_KEY(또는 GOOGLE_API_KEY). (pip install openai 필요)
        self.gemini_base_url = self.config.get(
            "base_url", "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.gemini_api_key = (
            self.config.get("_direct_api_key")
            or os.environ.get(self.config.get("api_key_env", "GEMINI_API_KEY"), "")
            or os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", "")
        )
        # (gemini 모델 티어는 _PROVIDER_DEFAULT_MODELS["gemini"]에서 결정 — #64)

        # [2026-05-21] 프로세스 전역 HCX rate limit — config에서 min_call_interval_ms.
        # 여러 LLMClient 인스턴스가 *같은 limiter*를 공유 → 모든 HCX 호출 직렬 간격 보장.
        _interval_ms = float(self.config.get("min_call_interval_ms", 0) or 0)
        if _interval_ms > 0:
            _hcx_rate_limiter.configure(_interval_ms / 1000.0)

        # [2026-05-25] HTTP timeout 및 timeout 발생 시 retry 횟수.
        # 기존 60s 하드코딩 → 응답 평균 5~15초인데 60s 끝까지 기다림 + retry 없음 →
        # 일시 네트워크 지연/끊김에 매우 약함. 30s로 줄여 빠른 실패 + retry 1회로 회복.
        self.http_timeout = float(self.config.get("http_timeout_seconds", 30.0))
        self.timeout_max_retries = int(self.config.get("timeout_max_retries", 1))

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
        elif self.provider == "upstage":  # [v2 - 김예슬] HCX 키 없을 때 Upstage(Solar)
            return await self._call_upstage(prompt, system_prompt, temperature, model_tier)
        elif self.provider == "gemini":  # [v2 - 김예슬]
            return await self._call_gemini(prompt, system_prompt, temperature, model_tier)
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
            # ── [legacy] strict json_schema 직접 호출 (additionalProperties 400 시 폐기) ──
            # return await self._call_openai_structured(...)  아래 _call_openai_structured
            # legacy 본문(strict json_schema) 참고.
            # OpenAI: strict json_schema 대신 Upstage/Gemini와 동일하게
            # json_object + 프롬프트 스키마 힌트 (HCX canonical 스키마 공유 유지).
            return await self._call_openai_structured(prompt, schema, system_prompt)
        elif self.provider == "upstage":  # [v2 - 김예슬]
            return await self._call_upstage_structured(prompt, schema, system_prompt)
        elif self.provider == "gemini":  # [v2 - 김예슬]
            return await self._call_gemini_structured(prompt, schema, system_prompt)
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

        [박재윤 - 2026-05-14]
        - 429 rate limit 시 exponential backoff 재시도 (최대 3회)
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

        # [박재윤 - 2026-05-14]: 429 exponential backoff (1초 → 2초 → 4초)
        # [2026-05-21] 매 시도 직전 전역 rate limiter 통과 — phase 간 burst 차단.
        # [2026-05-25] timeout도 retry — 일시 네트워크 지연에 회복하도록.
        max_retries = 3
        timeout_attempts = 0  # timeout 누적 횟수 (429와 분리)
        for attempt in range(max_retries):
            try:
                await _hcx_rate_limiter.acquire()
                async with httpx.AsyncClient(timeout=self.http_timeout) as client:
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
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    # [2026-05-21] jitter 추가 — 동시 N개 호출이 모두 429 받으면 같은 시점에
                    # retry해 또 burst 발생. 0~0.7s 랜덤 지연으로 동기화 깸.
                    import random as _random
                    wait = (2 ** attempt) + _random.uniform(0.0, 0.7)
                    # [2026-05-21] HCX 응답 헤더에 rate limit 정보 있는지 한 번만 로그.
                    # X-RateLimit-* / Retry-After 등이 있으면 한도 추정 가능.
                    _hcx_rate_limiter._log_headers_if_first(e.response.headers)
                    logger.warning(f"HCX v1 429 rate limit — {wait:.2f}초 후 재시도 ({attempt+1}/{max_retries})")
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"HCX v1 HTTP 오류: {e.response.status_code} — {e.response.text[:200]}")
                    raise
            except httpx.TimeoutException:
                if timeout_attempts < self.timeout_max_retries:
                    timeout_attempts += 1
                    import random as _random
                    wait = 0.5 + _random.uniform(0.0, 0.5)
                    logger.warning(
                        f"HCX v1 타임아웃 ({self.http_timeout}s) — "
                        f"{wait:.2f}초 후 재시도 ({timeout_attempts}/{self.timeout_max_retries})"
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error(f"HCX v1 타임아웃 (재시도 소진): {url}")
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

        [박재윤 - 2026-05-14]
        - 429 rate limit 시 exponential backoff 재시도 (최대 3회)
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

        # [박재윤 - 2026-05-14]: 429 exponential backoff (1초 → 2초 → 4초)
        # [2026-05-21] 매 시도 직전 전역 rate limiter 통과 — phase 간 burst 차단.
        # [2026-05-25] timeout도 retry.
        max_retries = 3
        timeout_attempts = 0
        for attempt in range(max_retries):
            try:
                await _hcx_rate_limiter.acquire()
                async with httpx.AsyncClient(timeout=self.http_timeout) as client:
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
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    # [2026-05-21] jitter 추가 (동시 N개 retry burst 동기화 깸)
                    import random as _random
                    wait = (2 ** attempt) + _random.uniform(0.0, 0.7)
                    _hcx_rate_limiter._log_headers_if_first(e.response.headers)
                    logger.warning(f"HCX v3 429 rate limit — {wait:.2f}초 후 재시도 ({attempt+1}/{max_retries})")
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"HCX v3 HTTP 오류: {e.response.status_code} — {e.response.text[:200]}")
                    raise
            except httpx.TimeoutException:
                if timeout_attempts < self.timeout_max_retries:
                    timeout_attempts += 1
                    import random as _random
                    wait = 0.5 + _random.uniform(0.0, 0.5)
                    logger.warning(
                        f"HCX v3 타임아웃 ({self.http_timeout}s) — "
                        f"{wait:.2f}초 후 재시도 ({timeout_attempts}/{self.timeout_max_retries})"
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error(f"HCX v3 타임아웃 (재시도 소진): {url}")
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

        [박재윤 - 2026-05-18]: 429 retry backoff 추가
        - _call_hcx_v1, _call_hcx_v3와 동일한 exponential backoff 적용 (최대 3회)
        """
        import asyncio

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
            "thinking": {"effort": "none"},
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

        max_retries = 3
        timeout_attempts = 0
        # [2026-05-21] 전역 rate limiter 통과 — phase 간 burst 차단
        # [2026-05-25] timeout retry 추가
        for attempt in range(max_retries):
            try:
                await _hcx_rate_limiter.acquire()
                async with httpx.AsyncClient(timeout=self.http_timeout) as client:
                    resp = await client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()

                status_code = data.get("status", {}).get("code", "")
                if status_code != "20000":
                    msg = data.get("status", {}).get("message", "unknown")
                    raise RuntimeError(f"HCX Structured Outputs 오류: {status_code} {msg}")

                content = data["result"]["message"]["content"]
                logger.debug(f"HCX Structured 응답: {content[:80]}...")
                return json.loads(content)

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    # [2026-05-21] jitter 추가 (동시 N개 retry burst 동기화 깸)
                    import random as _random
                    wait = (2 ** attempt) + _random.uniform(0.0, 0.7)
                    _hcx_rate_limiter._log_headers_if_first(e.response.headers)
                    logger.warning(
                        f"HCX Structured 429 rate limit — {wait:.2f}초 후 재시도 "
                        f"({attempt+1}/{max_retries})"
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        f"HCX Structured HTTP 오류: {e.response.status_code} — "
                        f"{e.response.text[:200]}"
                    )
                    raise
            except httpx.TimeoutException:
                if timeout_attempts < self.timeout_max_retries:
                    timeout_attempts += 1
                    import random as _random
                    wait = 0.5 + _random.uniform(0.0, 0.5)
                    logger.warning(
                        f"HCX Structured 타임아웃 ({self.http_timeout}s) — "
                        f"{wait:.2f}초 후 재시도 ({timeout_attempts}/{self.timeout_max_retries})"
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error(f"HCX Structured 타임아웃 (재시도 소진): {url}")
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

        model = self.models.get(model_tier, self.default_model)

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
        """OpenAI 구조화 응답 — json_object + 스키마 프롬프트 힌트 (strict json_schema 미사용)."""
        # ── [legacy] OpenAI strict json_schema (프롬프트 힌트 없음) ─────────────
        # canonical 스키마에 additionalProperties 등 strict 규칙 맞춘 뒤 재활성 검토.
        # """OpenAI JSON Schema 구조화 응답"""
        # try:
        #     from openai import AsyncOpenAI
        # except ImportError:
        #     raise ImportError("pip install openai")
        #
        # messages = []
        # if system_prompt:
        #     messages.append({"role": "system", "content": system_prompt})
        # messages.append({"role": "user", "content": prompt})
        #
        # model = self.models.get("structured", self.default_model)
        # client = AsyncOpenAI(api_key=self.openai_api_key)
        # resp = await client.chat.completions.create(
        #     model=model,
        #     messages=messages,
        #     temperature=self.temperature,
        #     max_tokens=self.max_tokens,
        #     response_format={"type": "json_schema", "json_schema": {
        #         "name": "structured_output", "strict": True, "schema": schema
        #     }},
        # )
        # content = resp.choices[0].message.content
        # return json.loads(content)

        base_url = self.config.get("base_url") or "https://api.openai.com/v1"
        return await self._call_openai_compatible_structured(
            prompt,
            schema,
            system_prompt,
            base_url=base_url,
            api_key=self.openai_api_key,
        )

    # ── [v2 - 김예슬] OpenAI 호환 provider (upstage·gemini) — base_url만 다름 ──

    async def _call_openai_compatible(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float | None,
        model_tier: str,
        *,
        base_url: str,
        api_key: str,
    ) -> str:
        """OpenAI 호환 Chat Completions 공통 호출 (upstage·gemini가 공유)."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("pip install openai")
        model = self.models.get(model_tier, self.default_model)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        resp = await client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content

    async def _call_openai_compatible_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_prompt: str | None,
        *,
        base_url: str,
        api_key: str,
    ) -> dict[str, Any]:
        """OpenAI 호환 구조화 응답 공통 호출.
        json_schema strict를 보장 못하는 provider가 있어, json_object 모드 + 스키마를
        프롬프트 힌트로 주고 파싱한다(response_format 미지원 시 일반 호출로 폴백).
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("pip install openai")
        sys_p = (system_prompt or "")
        sys_p += (
            "\n\n반드시 아래 JSON 스키마에 맞는 JSON 객체 하나만 출력하세요. "
            "설명·코드펜스 없이 JSON만 출력:\n"
            + json.dumps(schema, ensure_ascii=False)
        )
        messages = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": prompt},
        ]
        model = self.models.get("structured", self.default_model)
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        try:
            resp = await client.chat.completions.create(
                model=model, messages=messages,
                temperature=self.temperature, max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception:
            resp = await client.chat.completions.create(
                model=model, messages=messages,
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
        return _parse_json_response(resp.choices[0].message.content)

    # Upstage (Solar)
    async def _call_upstage(self, prompt, system_prompt, temperature, model_tier) -> str:
        return await self._call_openai_compatible(
            prompt, system_prompt, temperature, model_tier,
            base_url=self.upstage_base_url, api_key=self.upstage_api_key,
        )

    async def _call_upstage_structured(self, prompt, schema, system_prompt) -> dict[str, Any]:
        return await self._call_openai_compatible_structured(
            prompt, schema, system_prompt,
            base_url=self.upstage_base_url, api_key=self.upstage_api_key,
        )

    # Gemini (Google) — OpenAI 호환 엔드포인트
    async def _call_gemini(self, prompt, system_prompt, temperature, model_tier) -> str:
        return await self._call_openai_compatible(
            prompt, system_prompt, temperature, model_tier,
            base_url=self.gemini_base_url, api_key=self.gemini_api_key,
        )

    async def _call_gemini_structured(self, prompt, schema, system_prompt) -> dict[str, Any]:
        return await self._call_openai_compatible_structured(
            prompt, schema, system_prompt,
            base_url=self.gemini_base_url, api_key=self.gemini_api_key,
        )


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