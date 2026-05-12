"""
sv_platform.api.routes.datasources — /v1/datasources

Phase 1 (현재): kosis 하드코딩만 반환. 회사 데이터(CSV/DB) 등록은 Phase 4에서 구현.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from sv_platform.api.middleware.auth import AuthContext, get_auth


router = APIRouter(prefix="/v1/datasources", tags=["datasources"])


@router.get("")
async def list_datasources(ctx: AuthContext = Depends(get_auth)):
    """현재 tenant가 사용 가능한 datasource 목록.

    Phase 1: KOSIS만. Phase 4에서 tenant의 custom CSV/DB datasource 합산.
    """
    return [
        {
            "id": "kosis",
            "name": "KOSIS (공공)",
            "type": "kosis",
            "status": "ready",
            "row_count": None,
            "created_at": datetime(1970, 1, 1, tzinfo=timezone.utc).isoformat(),
        }
    ]