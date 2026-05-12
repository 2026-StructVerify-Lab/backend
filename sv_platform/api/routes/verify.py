"""
sv_platform.api.routes.verify — /v1/verify

Phase 1 (현재): synchronous 실행 — 요청 받자마자 라이브러리 호출,
                완료 시까지 대기 후 JobOut 통째로 반환.
Phase 2:        비동기 — submit하면 즉시 pending 반환, 워커가 백그라운드 처리.

[응답 모양]
프론트의 `lib/types.ts:Job`과 정확히 같은 모양 (JobOut) 반환.
즉 sync 모드에서도 첫 응답에 result까지 채워서 줌. 프론트는 polling 안 하고도
바로 결과 화면 그릴 수 있음 (그래도 page.tsx는 polling 로직 그대로 둠 — Phase 2 준비).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from sv_platform.api.middleware.auth import AuthContext, get_auth
from sv_platform.api.schemas.verify import JobOut, VerifyRequest
from sv_platform.db import get_session
from sv_platform.models.job import Job
from sv_platform.pipeline_runner import run_verification


router = APIRouter(prefix="/v1", tags=["verify"])
logger = logging.getLogger(__name__)


@router.post("/verify", response_model=JobOut)
async def submit_verify(
    req: VerifyRequest,
    ctx: AuthContext = Depends(get_auth),
    db: AsyncSession = Depends(get_session),
) -> JobOut:
    """
    검증 요청 제출.

    - text: source_data 필수
    - url:  url 필수 (Phase 3)
    - pdf/docx: multipart 업로드 별도 endpoint (Phase 3)

    Phase 1: 응답 시점에 이미 라이브러리 실행 끝나있음 → status="completed" 또는 "failed".
    Phase 2부터는 즉시 status="pending" 반환, 진짜 결과는 GET /v1/jobs/{id}.
    """
    # 입력 검증
    if req.source_type == "text" and not req.source_data:
        raise HTTPException(400, "source_data is required for text")
    if req.source_type == "url" and not req.url:
        raise HTTPException(400, "url is required for url type")
    if req.source_type in ("pdf", "docx"):
        raise HTTPException(
            400,
            "Use multipart upload endpoint (Phase 3 — coming soon)",
        )

    # Job 레코드 생성
    job = Job(
        tenant_id=ctx.tenant_id,
        api_key_id=ctx.api_key_id,
        status="running",
        source_type=req.source_type,
        source_data=req.source_data,
        datasources=req.datasources,
        callback_url=req.callback_url,
        started_at=datetime.now(timezone.utc),
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # ── Phase 1: 동기 실행 ──────────────────────────────────────
    try:
        result = await run_verification(
            source_type=req.source_type,
            source_data=req.source_data,
            datasources=req.datasources,
        )
        job.status = "completed"
        job.result = result
        job.progress = 100
        job.completed_at = datetime.now(timezone.utc)
    except NotImplementedError as e:
        job.status = "failed"
        job.error = str(e)
        job.completed_at = datetime.now(timezone.utc)
    except Exception as e:
        logger.exception("Pipeline execution failed for job %s", job.id)
        job.status = "failed"
        job.error = str(e)
        job.completed_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(job)

    # ⚠️ JobOut 통째로 반환 — 프론트의 Job 타입과 매칭됨
    # 키 이름이 `id` (job_id 아님), result/status까지 다 포함
    return JobOut.model_validate(job)