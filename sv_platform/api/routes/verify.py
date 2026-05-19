"""
sv_platform.api.routes.verify — /v1/verify

[Phase 2 - 비동기]
- 요청 받으면 즉시 job 생성 + BackgroundTask로 라이브러리 실행 → JobOut(status=pending) 반환
- 프론트는 GET /v1/jobs/{id} 폴링으로 진행 상황 확인
- BackgroundTask는 단일 uvicorn worker 내에서 asyncio task로 실행
"""
from __future__ import annotations

from structverify.utils.logger import get_logger
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from sv_platform.api.middleware.auth import AuthContext, get_auth
from sv_platform.api.schemas.verify import JobOut, VerifyRequest
from sv_platform.db import get_session
from sv_platform.models.job import Job
from sv_platform.pipeline_runner import run_verification_background


router = APIRouter(prefix="/v1", tags=["verify"])
logger = get_logger(__name__)


@router.post("/verify", response_model=JobOut)
async def submit_verify(
    req: VerifyRequest,
    background_tasks: BackgroundTasks,
    ctx: AuthContext = Depends(get_auth),
    db: AsyncSession = Depends(get_session),
) -> JobOut:
    """
    검증 요청 제출 — 즉시 job_id 반환, 백그라운드에서 처리.

    응답 시점: status='pending', progress=0
    완료 후: GET /v1/jobs/{id}로 확인 (프론트 폴링)
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

    # Job 레코드 생성 (status='pending')
    job = Job(
        tenant_id=ctx.tenant_id,
        api_key_id=ctx.api_key_id,
        status="pending",
        source_type=req.source_type,
        source_data=req.source_data,
        source_uri=req.url,
        datasources=req.datasources,
        callback_url=req.callback_url,
        progress=0,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # 백그라운드 task 시작 (response 전송 후 실행됨)
    background_tasks.add_task(
        run_verification_background,
        job_id=job.id,
        source_type=req.source_type,
        source_data=req.source_data,
        source_uri=req.url,
        datasources=req.datasources,
    )

    logger.info(f"Verify submitted: job={job.id}")

    # 즉시 응답 — 프론트는 폴링으로 진행 상황 확인
    return JobOut.model_validate(job)