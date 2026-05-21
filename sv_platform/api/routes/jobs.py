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
    """Job 상세. 다른 tenant의 job 접근 시 404 (정보 누설 방지).

    [실시간 plan/trace 노출] 잡이 진행 중(status=pending/running)이면
    agent_workspace에서 partial plan/trace를 읽어 result에 채워 보냄.
    프론트는 폴링하면서 claim별 진행 상황을 progressively 렌더 가능.
    """
    job = await db.get(Job, job_id)
    if job is None or job.tenant_id != ctx.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    out = JobOut.model_validate(job)

    # 진행 중일 때 workspace에서 partial 데이터 enrich
    if job.status in ("pending", "running") and not out.result:
        try:
            from sv_platform.loaders.workspace_reader import (
                read_partial_job_workspace,
            )
            # source_text 같이 넘김 — 라이브러리가 자체 doc_id로 워크스페이스를
            # 만든 경우 sv_platform job_id로 정확 매칭이 안 됨. job.source_data
            # 와 workspace/<>/source.txt 일치 검사로 정확한 디렉토리 찾음.
            # created_after = job.created_at — workspace_dir 재사용 버그로
            # 과거 잡들의 claim이 같은 디렉토리에 누적될 수 있음. 현재 잡의
            # created_at 이후 mtime의 claim만 골라 응답.
            created_after_ts = (
                job.created_at.timestamp() if job.created_at else None
            )
            partial = read_partial_job_workspace(
                str(job_id),
                source_text=job.source_data,
                created_after=created_after_ts,
            )
            if partial:
                out.result = partial
                # ★ library logger의 propagate=False 때문에 sv_platform이
                # 부모 로거에 붙인 progress handler가 안 깨어남.
                # partial.claims 상태에서 진행률을 역추정해 화면에 반영.
                _enriched_progress, _enriched_step = _estimate_progress_from_partial(
                    partial
                )
                if _enriched_progress > out.progress:
                    out.progress = _enriched_progress
                    out.current_step = _enriched_step
        except Exception:
            # 워크스페이스 읽기 실패해도 폴링 응답 자체는 막지 않음
            pass

    return out


def _estimate_progress_from_partial(
    partial: dict,
) -> tuple[int, str]:
    """partial.claims 상태로 진행률 추정 (DB의 progress가 5%에 멈춰 있을 때 사용).

    구간:
      - claims 자체가 없음 → 30%, "검증 가능 주장 탐지"
      - claim에 plan만 있음 → 50%, "Claim 그래프 빌드"
      - claim에 trace 일부 → 50~80%, "공식 통계 조회"/"수치 검증"
      - 모든 claim에 verdict → 95%, "근거 설명 생성"

    DB 값보다 큰 경우에만 caller가 덮어쓰도록 단순 튜플 반환.
    """
    claims = (partial or {}).get("claims") or []
    total = len(claims)
    if total == 0:
        return 30, "검증 가능 주장 탐지"
    with_verdict = sum(1 for c in claims if c.get("verdict"))
    with_trace = sum(1 for c in claims if c.get("trace"))
    with_plan = sum(1 for c in claims if c.get("plan"))
    if with_verdict == total:
        return 95, "근거 설명 생성"
    if with_verdict > 0:
        # 65~90% 사이 — verdict 비율에 비례
        return 65 + int(25 * with_verdict / total), "수치 검증"
    if with_trace > 0:
        # agent loop 진행 중
        return 50 + int(15 * with_trace / total), "공식 통계 조회"
    if with_plan > 0:
        return 50, "Claim 그래프 빌드"
    return 40, "스키마 유도"


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
