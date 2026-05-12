"""
sv_platform.models — ORM 모델 모음

Alembic이 마이그레이션 생성 시 이 패키지의 모든 모델 metadata를 사용.
새 모델 추가 시 여기에 import 추가하는 것 잊지 말 것 (안 그러면 Alembic이 못 봄).
"""
from sv_platform.models.base import Base
from sv_platform.models.tenant import Tenant
from sv_platform.models.user import User
from sv_platform.models.api_key import ApiKey
from sv_platform.models.job import Job

__all__ = ["Base", "Tenant", "User", "ApiKey", "Job"]
