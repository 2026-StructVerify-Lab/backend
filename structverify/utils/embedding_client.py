"""structverify.utils.embedding_client — 공용 임베딩 클라이언트 (#67-A 골격).

config.embedding({provider, model, api_key_env, base_url?})을 읽어 provider 분기.
utils/llm_client.py 의 provider dispatch 패턴을 따름.
이번 단계는 *골격만* — 실제 HTTP 호출(_embed_hcx 등)은 NotImplementedError 스텁.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from structverify.utils.logger import get_logger

logger = get_logger(__name__)

EMBEDDING_DIM = 1024  # HCX-EMB-V2 차원 (memory/embedder.py:26 과 동일)

# HCX 호출 수치 — 일단 상수, 나중에 config화 여지 (kosis_crawler 기준)
_HCX_TIMEOUT = 30          # 초
_BATCH_CONCURRENCY = 3     # asyncio.Semaphore 동시 호출 수
_BATCH_MAX_RETRY = 5       # 429 등 재시도 횟수

# provider별 기본값 (config 미지정 시). base_url은 다음 단계 호출에서 사용.
_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "hcx": {
        "api_key_env": "CLOVASTUDIO_API_KEY",
        "base_url": "https://clovastudio.stream.ntruss.com/v1/api-tools/embedding/v2",
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model": "text-embedding-3-small",
    },
    "upstage": {
        "api_key_env": "UPSTAGE_API_KEY",
        "base_url": "https://api.upstage.ai/v1",
        "model": "solar-embedding-1-large",
    },
}


def _load_async_openai():
    """openai SDK AsyncOpenAI 로드. 없으면 None(httpx 폴백). 테스트 monkeypatch 지점."""
    try:
        from openai import AsyncOpenAI
        return AsyncOpenAI
    except ImportError:
        return None


class EmbeddingClient:
    """텍스트 → 임베딩 벡터. provider(hcx/openai/upstage) 분기.

    config 예 (config/default.yaml의 embedding 섹션):
        provider: "hcx"
        model: "HCX-EMB-V2"
        api_key_env: "NCP_API_KEY"
        base_url: "..."            # 선택 (provider 기본값 있음)
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        # provider 디폴트는 두지 않되, 없으면 "hcx" (기존 동작 보존)
        self.provider = self.config.get("provider", "hcx")
        _defaults = _PROVIDER_DEFAULTS.get(self.provider, {})
        self.model = self.config.get("model") or _defaults.get("model")
        # 키는 *값*이 아니라 api_key_env *이름*으로 보관 → 호출 시점에 os.environ 조회 (lazy)
        self._api_key_env = self.config.get("api_key_env") or _defaults.get("api_key_env", "")
        self.base_url = self.config.get("base_url") or _defaults.get("base_url", "")
        logger.info(
            f"[EmbeddingClient] provider={self.provider}, model={self.model!r}, "
            f"api_key_env={self._api_key_env!r}"
        )

    @property
    def api_key(self) -> str:
        """api_key_env가 가리키는 환경변수 값. 없으면 "" (예외 던지지 않음, lazy)."""
        if not self._api_key_env:
            return ""
        return os.environ.get(self._api_key_env, "")

    # ── 공개 인터페이스 ──

    async def embed(self, text: str) -> list[float] | None:
        """단건 임베딩 — provider 분기."""
        if self.provider == "hcx":
            return await self._embed_hcx(text)
        elif self.provider == "openai":
            return await self._embed_openai(text)
        elif self.provider == "upstage":
            return await self._embed_upstage(text)
        raise ValueError(f"미지원 embedding provider: {self.provider}")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """배치 임베딩 — provider 분기."""
        if self.provider == "hcx":
            return await self._embed_batch_hcx(texts)
        elif self.provider == "openai":
            return await self._embed_batch_openai(texts)
        elif self.provider == "upstage":
            return await self._embed_batch_upstage(texts)
        raise ValueError(f"미지원 embedding provider: {self.provider}")

    # ── provider별 실제 호출 (다음 단계 구현) ──

    async def _embed_hcx(self, text: str) -> list[float] | None:
        # catalog_search._get_embedding 이식. 키 없으면 None, 예외 → 로그 후 None.
        if not self.api_key:
            return None
        try:
            async with httpx.AsyncClient(timeout=_HCX_TIMEOUT) as client:
                resp = await client.post(
                    self.base_url,  # config base_url 우선 (기본=hcx 엔드포인트)
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"text": text},
                )
                return resp.json()["result"]["embedding"]
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[EmbeddingClient] hcx 임베딩 실패: {e}")
            return None

    async def _embed_openai(self, text: str) -> list[float] | None:
        res = await self._embed_openai_compatible([text])
        return res[0] if res else None

    async def _embed_upstage(self, text: str) -> list[float] | None:
        # upstage도 OpenAI 호환 → 같은 경로 (base_url/model은 self에 provider별 반영됨)
        res = await self._embed_openai_compatible([text])
        return res[0] if res else None

    async def _embed_openai_compatible(
        self, inputs: list[str]
    ) -> list[list[float]] | None:
        """openai·upstage 공유. inputs(list) → 임베딩 list. 키 없으면 None, 예외→로그→None.

        dimensions=EMBEDDING_DIM(1024) 강제 — pgvector vector(1024) 고정에 맞춤.
        ⚠️ upstage(solar-embedding-1-large)는 dimensions 미지원일 수 있음 → 원차원(예: 4096)
           반환 가능. 그 경우 아래 길이 검증에서 경고하고 그대로 반환(호출부/DB가 불일치 판단:
           pgvector vector(1024)에 INSERT 시 차원 안 맞으면 거부됨).
        """
        if not self.api_key:
            return None
        try:
            _AsyncOpenAI = _load_async_openai()
            if _AsyncOpenAI is not None:
                # OpenAI 공식 SDK 경로
                client = _AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
                resp = await client.embeddings.create(
                    model=self.model, input=inputs, dimensions=EMBEDDING_DIM,
                )
                vectors = [d.embedding for d in resp.data]
            else:
                # SDK 없으면 httpx 로 POST {base_url}/embeddings
                async with httpx.AsyncClient(timeout=_HCX_TIMEOUT) as client:
                    resp = await client.post(
                        f"{self.base_url.rstrip('/')}/embeddings",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model,
                            "input": inputs,
                            "dimensions": EMBEDDING_DIM,
                        },
                    )
                    data = resp.json()
                    vectors = [d["embedding"] for d in data["data"]]
            if vectors and len(vectors[0]) != EMBEDDING_DIM:
                logger.warning(
                    f"[EmbeddingClient] {self.provider} 임베딩 차원 {len(vectors[0])} "
                    f"≠ EMBEDDING_DIM({EMBEDDING_DIM}) — pgvector(1024) 불일치 위험"
                )
            return vectors
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[EmbeddingClient] {self.provider} 임베딩 실패: {e}")
            return None

    async def _embed_batch_hcx(self, texts: list[str]) -> list[list[float]]:
        # kosis_crawler.get_embedding_safe 이식: Semaphore + 429 지수백오프 + 제로벡터 폴백.
        if not self.api_key:
            logger.warning("[EmbeddingClient] api_key 없음 → batch 제로벡터 반환")
            return [[0.0] * EMBEDDING_DIM for _ in texts]
        sem = asyncio.Semaphore(_BATCH_CONCURRENCY)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async def _one(client: httpx.AsyncClient, text: str) -> list[float]:
            async with sem:
                for retry in range(_BATCH_MAX_RETRY):
                    resp = await client.post(
                        self.base_url, headers=headers, json={"text": text},
                    )
                    data = resp.json()
                    if data.get("result") is not None:
                        return data["result"]["embedding"]
                    if resp.status_code == 429:
                        await asyncio.sleep(2 ** retry)
                return [0.0] * EMBEDDING_DIM

        async with httpx.AsyncClient(timeout=_HCX_TIMEOUT) as client:
            return list(await asyncio.gather(*[_one(client, t) for t in texts]))

    async def _embed_batch_openai(self, texts: list[str]) -> list[list[float]]:
        # OpenAI는 input=list 배치 지원 → 한 번 호출. 실패 시 제로벡터 폴백(hcx와 동일 규약).
        res = await self._embed_openai_compatible(texts)
        return res if res is not None else [[0.0] * EMBEDDING_DIM for _ in texts]

    async def _embed_batch_upstage(self, texts: list[str]) -> list[list[float]]:
        res = await self._embed_openai_compatible(texts)
        return res if res is not None else [[0.0] * EMBEDDING_DIM for _ in texts]
