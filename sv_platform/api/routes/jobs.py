"""
sv_platform.api.routes.jobs — /v1/jobs

- GET /v1/jobs/{id}   : 단일 job 상세 (status + progress + result)
- GET /v1/jobs        : 사용자 tenant의 job 목록 (페이지네이션)
"""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sv_platform.api.middleware.auth import AuthContext, get_auth
from sv_platform.api.schemas.verify import JobOut
from sv_platform.db import get_session
from sv_platform.models.job import Job


router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobOut)
async def get_job(
    job_id: UUID,
    ctx: AuthContext = Depends(get_auth),
    db: AsyncSession = Depends(get_session),
) -> JobOut:
    """Job 상세. 다른 tenant의 job 접근 시 404 (정보 누설 방지)."""
    job = await db.get(Job, job_id)
    if job is None or job.tenant_id != ctx.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    return JobOut.model_validate(job)


@router.get("", response_model=list[JobOut])
async def list_jobs(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    ctx: AuthContext = Depends(get_auth),
    db: AsyncSession = Depends(get_session),
) -> list[JobOut]:
    """현재 tenant의 최근 job 목록 (최신 순)."""
    stmt = (
        select(Job)
        .where(Job.tenant_id == ctx.tenant_id)
        .order_by(Job.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(stmt)
    jobs = result.scalars().all()
    return [JobOut.model_validate(j) for j in jobs]
