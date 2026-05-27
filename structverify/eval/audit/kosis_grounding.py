"""Verify evidence stat_table_id resolves in KOSIS."""
from __future__ import annotations

import os
import re
from typing import Any

import httpx

from structverify.core.schemas import VerificationResult
from structverify.retrieval.kosis_connector import kosis_get_meta
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

# DT_{org}_{tbl} style ids used in catalog (multi-segment tbl)
_DT_RE = re.compile(r"^DT_([^_]+)_(.+)$", re.IGNORECASE)


_DEFAULT_KOSIS_BASE = "https://kosis.kr/openapi"


def normalize_kosis_base_url(base_url: str | None) -> str:
    """Align with KOSISConnector: openapi root only (no statisticsData.do suffix)."""
    base = (base_url or _DEFAULT_KOSIS_BASE).rstrip("/")
    suffix = "/statisticsdata.do"
    if base.lower().endswith(suffix):
        base = base[: -len(suffix)].rstrip("/")
    return base


def _format_grounding_error(meta: dict[str, Any]) -> str:
    err = meta.get("errMsg") or meta.get("error") or meta.get("kosis_error") or meta.get("err")
    if err is None:
        return "unknown"
    parts = [str(err)]
    detail = meta.get("detail")
    if detail and str(detail) not in parts[0]:
        parts.append(str(detail)[:300])
    api_err = meta.get("err")
    if api_err is not None and str(api_err) not in parts[0]:
        parts.append(f"err={api_err}")
    return ": ".join(parts)


def resolve_kosis_api_key(kosis_cfg: dict[str, Any] | None) -> str | None:
    """Match KOSISConnector: explicit api_key or env via api_key_env."""
    cfg = kosis_cfg or {}
    key = cfg.get("api_key")
    if key:
        return str(key).strip() or None
    env_name = cfg.get("api_key_env") or "KOSIS_API_KEY"
    return (os.environ.get(env_name) or "").strip() or None


def resolve_org_and_tbl(
    stat_id: str | None,
    *,
    org_id_hint: str | None = None,
) -> tuple[str, str] | None:
    """Return (orgId, tblId) for KOSIS getMeta.

    Production fetch uses orgId from catalog and tblId = full stat_id (e.g. DT_200Y108).
    When parsing fails, org_id_hint from the outcome case (kosis_org_id) is used with
    the full stat_id as tblId.
    """
    if not stat_id:
        return None
    s = stat_id.strip()
    hint = (org_id_hint or "").strip()
    if hint:
        return hint, s

    m = _DT_RE.match(s)
    if m:
        return m.group(1), m.group(2)
    if "_" in s and s.upper().startswith("DT_"):
        parts = s.split("_", 2)
        if len(parts) >= 3:
            return parts[1], parts[2]
    return None


def parse_stat_table_id(
    stat_id: str | None,
    *,
    org_id_hint: str | None = None,
) -> tuple[str, str] | None:
    """Backward-compatible alias for resolve_org_and_tbl."""
    return resolve_org_and_tbl(stat_id, org_id_hint=org_id_hint)


async def check_kosis_grounding(
    result: VerificationResult,
    *,
    api_key: str | None,
    base_url: str,
    timeout: float = 30.0,
    org_id_hint: str | None = None,
) -> dict[str, Any]:
    """Check whether evidence stat_table_id exists via KOSIS getMeta."""
    out: dict[str, Any] = {
        "kosis_grounding_checked": False,
        "kosis_grounding_ok": None,
        "stat_table_id": None,
        "grounding_error": None,
    }
    ev = result.evidence
    if not ev or not ev.stat_table_id:
        return out

    stat_id = ev.stat_table_id
    out["stat_table_id"] = stat_id
    parsed = resolve_org_and_tbl(stat_id, org_id_hint=org_id_hint)
    if not parsed:
        out["kosis_grounding_checked"] = True
        out["kosis_grounding_ok"] = False
        out["grounding_error"] = "unparseable_stat_id"
        return out

    if not api_key:
        out["grounding_error"] = "no_api_key"
        return out

    org_id, tbl_id = parsed
    out["kosis_grounding_checked"] = True
    base = normalize_kosis_base_url(base_url)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            meta = await kosis_get_meta(
                client, base, api_key, org_id, tbl_id, "PRD", timeout
            )
        if isinstance(meta, dict) and (
            meta.get("kosis_error") or meta.get("error") or meta.get("err") is not None
        ):
            out["kosis_grounding_ok"] = False
            out["grounding_error"] = _format_grounding_error(meta)
        elif meta:
            out["kosis_grounding_ok"] = True
        else:
            out["kosis_grounding_ok"] = False
            out["grounding_error"] = "empty_meta"
    except httpx.HTTPError as e:
        out["kosis_grounding_ok"] = False
        out["grounding_error"] = f"http:{e}"
    except Exception as e:
        out["kosis_grounding_ok"] = False
        out["grounding_error"] = str(e)
    return out


async def grounding_from_config(
    result: VerificationResult,
    config: dict[str, Any],
    *,
    org_id_hint: str | None = None,
) -> dict[str, Any]:
    kosis = config.get("kosis") or {}
    api_key = resolve_kosis_api_key(kosis)
    base_url = normalize_kosis_base_url(kosis.get("base_url"))
    timeout = float(kosis.get("timeout", 30))
    return await check_kosis_grounding(
        result,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        org_id_hint=org_id_hint,
    )
