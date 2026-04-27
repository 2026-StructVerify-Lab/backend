"""
adaptation/adapter_trainer.py — NCP CLOVA Studio Tuning API 기반 학습 관리

HCX 모델은 로컬 PEFT가 아니라 NCP CLOVA Studio Tuning API를 사용합니다.
학습 데이터 → NCP Object Storage 업로드 → Tuning API 호출 → polling → 평가 → 배포

[김예슬 - 2026-04-24]
- train():
  · 학습 샘플 → NCP Tuning API JSONL 포맷 변환 (_samples_to_jsonl)
  · NCP Object Storage 업로드 (_upload_to_object_storage)
  · POST /tuning/v2/tasks 호출 (_call_tuning_api)
  · 완료까지 polling (_poll_tuning_status)
- evaluate():
  · 벤치마크 JSONL 로드 → Tuning된 모델로 추론 → F1 계산
- deploy():
  · model_versions 테이블 INSERT (MLflow는 추후 연동)
  · config 파일 갱신으로 runtime_agent에 핫스왑 알림
- _samples_to_jsonl(): NCP PEFT 학습 포맷으로 변환
  · {"text": "### 지시: ...\n\n### 질문: ...\n\n### 답변: ..."}

[NCP Tuning API]
  POST https://clovastudio.apigw.ntruss.com/tuning/v2/tasks
  tuningType: "PEFT"
  taskType: "GENERATION"

[참고] KnowLA (NAACL 2024), AdaptLLM (ICLR 2024)
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import httpx

from structverify.utils.logger import get_logger

logger = get_logger(__name__)

NCP_TUNING_URL        = "https://clovastudio.apigw.ntruss.com/tuning/v2/tasks"
NCP_TUNING_STATUS_URL = "https://clovastudio.apigw.ntruss.com/tuning/v2/tasks/{task_id}"
NCP_OBS_ENDPOINT      = "https://kr.object.ncloudstorage.com"


class AdapterTrainer:
    """NCP CLOVA Studio Tuning API 기반 Adapter 학습 관리자."""

    def __init__(self, config: dict | None = None):
        self.config    = config or {}
        adapt_cfg      = self.config.get("adaptation", {})
        self.eval_min_score = float(adapt_cfg.get("eval_min_score", 0.85))
        self.epochs    = int(adapt_cfg.get("train_epochs", 8))
        self.lr        = float(adapt_cfg.get("learning_rate", 1e-5))
        self.bucket    = self.config.get("storage", {}).get("bucket", "structverify-training")

        self.api_key        = os.environ.get("CLOVASTUDIO_API_KEY", "")
        self.ncp_access_key = os.environ.get("NCP_ACCESS_KEY", "")
        self.ncp_secret_key = os.environ.get("NCP_SECRET_KEY", "")

    async def train(self, domain: str, samples: list[dict[str, Any]]) -> str | None:
        """
        학습 샘플 → NCP Tuning API → adapter 반환.

        흐름:
          1) JSONL 변환
          2) NCP Object Storage 업로드
          3) Tuning API 호출
          4) 완료 polling
        """
        if not samples:
            logger.warning("학습 샘플 없음 — train() 중단")
            return None

        logger.info(f"[Trainer] 학습 시작: domain={domain}, samples={len(samples)}")

        # 1) JSONL 변환
        jsonl_path = f"/tmp/structverify_train_{domain}_{int(time.time())}.jsonl"
        _samples_to_jsonl(samples, jsonl_path)
        logger.info(f"[Trainer] JSONL 변환: {jsonl_path}")

        # 2) Object Storage 업로드
        remote_path = await _upload_to_object_storage(
            local_path=jsonl_path,
            bucket=self.bucket,
            access_key=self.ncp_access_key,
            secret_key=self.ncp_secret_key,
            domain=domain,
        )
        if not remote_path:
            logger.error("[Trainer] Object Storage 업로드 실패")
            return None

        # 3) Tuning API 호출
        task_name = f"structverify_{domain}_{int(time.time())}"
        task_id = await _call_tuning_api(
            api_key=self.api_key,
            task_name=task_name,
            model="HCX-003",
            ncp_file_path=remote_path,
            bucket=self.bucket,
            access_key=self.ncp_access_key,
            secret_key=self.ncp_secret_key,
            epochs=self.epochs,
            learning_rate=self.lr,
        )
        if not task_id:
            logger.error("[Trainer] Tuning API 호출 실패")
            return None

        # 4) 완료 polling
        adapter_path = await _poll_tuning_status(self.api_key, task_id)
        if adapter_path:
            logger.info(f"[Trainer] 학습 완료: {adapter_path}")
        return adapter_path

    async def evaluate(self, adapter_path: str, benchmark: str) -> float:
        """
        학습된 Adapter 성능 평가.

        benchmark JSONL 로드 → 각 샘플에 대해 LLM 추론 → 정확도 계산.
        """
        if not os.path.exists(benchmark):
            logger.warning(f"[Trainer] 벤치마크 파일 없음: {benchmark} → 기본 점수 0.0")
            return 0.0

        # 벤치마크 로드
        samples: list[dict] = []
        with open(benchmark, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if not samples:
            logger.warning("[Trainer] 벤치마크 샘플 없음")
            return 0.0

        # TODO: adapter_path 모델로 실제 추론 후 정확도 계산
        # 현재는 샘플 개수 기반 더미 점수
        # 실제 구현 시:
        #   from structverify.utils.llm_client import LLMClient
        #   llm = LLMClient(config={"adapter_path": adapter_path, ...})
        #   correct = 0
        #   for s in samples:
        #       pred = await llm.generate_json(s["input"])
        #       if pred.get("label") == s["expected_label"]:
        #           correct += 1
        #   return correct / len(samples)
        logger.warning(f"[Trainer] evaluate() stub: {len(samples)}개 샘플 → 0.0")
        return 0.0

    async def deploy(self, adapter_path: str, domain: str) -> bool:
        """
        평가 통과 Adapter 배포.

        1) model_versions 테이블에 등록 (TODO 박재윤)
        2) domain-packs/{domain}/model.yaml 업데이트
           → runtime_agent가 다음 요청부터 새 adapter 사용
        """
        logger.info(f"[Trainer] Adapter 배포: {adapter_path} → {domain}")

        # domain-packs/{domain}/ 디렉토리에 model 정보 기록
        pack_dir  = os.path.join("domain-packs", domain)
        model_yaml = os.path.join(pack_dir, "model.yaml")
        os.makedirs(pack_dir, exist_ok=True)

        import yaml
        model_info = {
            "adapter_path":  adapter_path,
            "domain":        domain,
            "deployed_at":   time.strftime("%Y-%m-%dT%H:%M:%S"),
            "base_model":    "HCX-003",
            "tuning_type":   "PEFT",
        }
        with open(model_yaml, "w", encoding="utf-8") as f:
            yaml.dump(model_info, f, allow_unicode=True)

        logger.info(f"[Trainer] 배포 완료: {model_yaml}")

        # TODO [박재윤]: model_versions 테이블 INSERT
        # await db.execute(
        #     "INSERT INTO model_versions (domain, adapter_path, deployed_at) VALUES (...)"
        # )

        return True


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────

def _samples_to_jsonl(samples: list[dict[str, Any]], output_path: str) -> None:
    """
    학습 샘플 → NCP PEFT 학습 포맷 JSONL 변환.

    NCP Tuning API 학습 포맷:
      {"text": "### 지시: {instruction}\n\n### 질문: {input}\n\n### 답변: {output}"}
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            instruction = sample.get("instruction", "")
            inp         = sample.get("input", "")
            out         = sample.get("output", "")

            if instruction:
                text = f"### 지시: {instruction}\n\n### 질문: {inp}\n\n### 답변: {out}"
            else:
                text = f"### 질문: {inp}\n\n### 답변: {out}"

            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


async def _upload_to_object_storage(
    local_path: str,
    bucket: str,
    access_key: str,
    secret_key: str,
    domain: str,
) -> str | None:
    """
    NCP Object Storage (S3 호환)에 학습 데이터 업로드.

    NCP Object Storage는 S3 호환 API 사용:
      endpoint: https://kr.object.ncloudstorage.com
    """
    if not access_key or not secret_key:
        logger.warning("[Trainer] NCP_ACCESS_KEY / NCP_SECRET_KEY 미설정 → stub")
        return f"training/{domain}/{os.path.basename(local_path)}"

    try:
        import boto3
        s3 = boto3.client(
            "s3",
            endpoint_url=NCP_OBS_ENDPOINT,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        remote_path = f"training/{domain}/{os.path.basename(local_path)}"
        s3.upload_file(local_path, bucket, remote_path)
        logger.info(f"[Trainer] Object Storage 업로드: s3://{bucket}/{remote_path}")
        return remote_path
    except Exception as e:
        logger.error(f"[Trainer] Object Storage 업로드 실패: {e}")
        return None


async def _call_tuning_api(
    api_key: str,
    task_name: str,
    model: str,
    ncp_file_path: str,
    bucket: str,
    access_key: str,
    secret_key: str,
    epochs: int = 8,
    learning_rate: float = 1e-5,
) -> str | None:
    """
    NCP CLOVA Studio Tuning API 호출.

    POST https://clovastudio.apigw.ntruss.com/tuning/v2/tasks
    """
    if not api_key:
        logger.warning("[Trainer] CLOVASTUDIO_API_KEY 미설정 → stub")
        return f"stub_task_{int(time.time())}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-NCP-CLOVASTUDIO-REQUEST-ID": f"train-{int(time.time())}",
    }
    payload = {
        "name":                       task_name,
        "model":                      model,
        "tuningType":                 "PEFT",
        "taskType":                   "GENERATION",
        "trainEpochs":                str(epochs),
        "learningRate":               str(learning_rate),
        "trainingDatasetFilePath":    ncp_file_path,
        "trainingDatasetBucket":      bucket,
        "trainingDatasetAccessKey":   access_key,
        "trainingDatasetSecretKey":   secret_key,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(NCP_TUNING_URL, data=payload, headers=headers)
            resp.raise_for_status()
            data    = resp.json()
            task_id = data.get("result", {}).get("taskId")
            logger.info(f"[Trainer] Tuning 작업 생성: task_id={task_id}")
            return task_id
    except Exception as e:
        logger.error(f"[Trainer] Tuning API 호출 실패: {e}")
        return None


async def _poll_tuning_status(
    api_key: str,
    task_id: str,
    poll_interval: int = 60,
    max_wait: int = 7200,
) -> str | None:
    """
    Tuning 작업 완료까지 polling.

    상태: READY / RUNNING / SUCCEEDED / FAILED / CANCELED
    """
    if task_id.startswith("stub_"):
        logger.warning(f"[Trainer] stub task_id — polling 스킵")
        return None

    headers = {"Authorization": f"Bearer {api_key}"}
    url     = NCP_TUNING_STATUS_URL.format(task_id=task_id)
    elapsed = 0

    while elapsed < max_wait:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp   = await client.get(url, headers=headers)
                resp.raise_for_status()
                data   = resp.json()
                status = data.get("result", {}).get("status", "")
                logger.info(f"[Trainer] Tuning 상태: {task_id} → {status} ({elapsed}s)")

                if status == "SUCCEEDED":
                    return task_id
                if status in ("FAILED", "CANCELED"):
                    logger.error(f"[Trainer] Tuning 실패: {status}")
                    return None

        except Exception as e:
            logger.warning(f"[Trainer] 상태 조회 실패 (재시도): {e}")

        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    logger.error(f"[Trainer] Tuning 타임아웃: {max_wait}s 초과")
    return None