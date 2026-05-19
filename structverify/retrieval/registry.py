"""
structverify.retrieval.registry — DataSource 동적 등록.

회사가 *자체 데이터 소스*를 등록 가능:

    # company_package/sales_source.py
    from structverify.retrieval.base import BaseDataSource
    from structverify.retrieval.registry import register_datasource

    @register_datasource("my_sales_db")
    class MySalesDB(BaseDataSource):
        async def search_catalog(self, query, ...):
            ...
        async def fetch_evidence(self, candidate_id, ...):
            ...

    # config/default.yaml
    data_sources:
      enabled: ["my_sales_db"]
      my_sales_db:
        dsn: "postgresql://..."

Agent loop (Phase D)에서 이 registry를 통해 source 인스턴스 생성:

    from structverify.retrieval.registry import build_datasource
    source = build_datasource("my_sales_db", config={"dsn": "..."})
"""
from __future__ import annotations

from structverify.utils.logger import get_logger
from typing import Callable, Type

from .base import BaseDataSource

logger = get_logger(__name__)


# ── Registry 본체 ────────────────────────────────────────────────

_REGISTRY: dict[str, Type[BaseDataSource]] = {}


def register_datasource(
    name: str,
) -> Callable[[Type[BaseDataSource]], Type[BaseDataSource]]:
    """
    DataSource 클래스를 등록하는 데코레이터.

    Args:
        name: config.data_sources.enabled 에 들어갈 이름.

    Example:
        @register_datasource("kosis")
        class KOSISSource(BaseDataSource):
            ...
    """
    def decorator(cls: Type[BaseDataSource]) -> Type[BaseDataSource]:
        if not issubclass(cls, BaseDataSource):
            raise TypeError(
                f"{cls.__name__} must subclass BaseDataSource to be registered"
            )
        if name in _REGISTRY:
            logger.warning(
                f"[registry] DataSource '{name}' already registered, overwriting "
                f"({_REGISTRY[name].__name__} → {cls.__name__})"
            )
        _REGISTRY[name] = cls
        cls.name = name
        logger.info(f"[registry] DataSource 등록: {name} ({cls.__name__})")
        return cls
    return decorator


def get_datasource_class(name: str) -> Type[BaseDataSource]:
    """등록된 클래스 반환. 없으면 KeyError."""
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(
            f"DataSource '{name}' not registered. "
            f"Available: {available}. "
            f"Register via @register_datasource('{name}')."
        )
    return _REGISTRY[name]


def list_datasources() -> list[str]:
    """등록된 모든 DataSource 이름."""
    return sorted(_REGISTRY.keys())


def build_datasource(name: str, config: dict | None = None) -> BaseDataSource:
    """Config에서 datasource 인스턴스 생성.

    Args:
        name: 등록된 이름.
        config: 클래스 __init__에 전달될 kwargs.
                None이면 인자 없이 생성.

    Example:
        source = build_datasource("kosis", config={"api_key": "..."})
    """
    cls = get_datasource_class(name)
    if config:
        return cls(**config)
    return cls()


# ── 편의 함수 ─────────────────────────────────────────────────────

def build_all_enabled(config: dict) -> list[BaseDataSource]:
    """
    config['data_sources'] 섹션에서 *enabled* 된 모든 source 빌드.

    Args:
        config: config.data_sources 섹션 전체 dict.
                예: {"enabled": ["kosis"], "kosis": {"api_key": "..."}, ...}

    Returns:
        활성화된 DataSource 인스턴스 리스트.
    """
    enabled = config.get("enabled", [])
    sources: list[BaseDataSource] = []
    for name in enabled:
        try:
            source_config = config.get(name, {})
            sources.append(build_datasource(name, source_config))
        except KeyError as e:
            logger.warning(
                f"[registry] enabled={name!r} but not registered — skipped. {e}"
            )
        except Exception as e:
            logger.error(f"[registry] build_datasource({name!r}) failed: {e}")
    return sources
