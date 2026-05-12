"""
sv_platform.pipeline_runner — `structverify` 라이브러리 호출 wrapper

라이브러리 report는 `claims`(메타) + `results`(검증 결과)가 *분리*돼 있어서
둘을 claim_id로 매칭해 합쳐야 verdict/evidence/explanation을 얻을 수 있음.
"""
from __future__ import annotations

import logging
from typing import Any

from sv_platform.config import settings

logger = logging.getLogger(__name__)


async def run_verification(
    source_type: str,
    source_data: str | None = None,
    source_uri: str | None = None,
    datasources: list[str] | None = None,
) -> dict[str, Any]:
    if source_type != "text":
        raise NotImplementedError(
            f"source_type='{source_type}'은 Phase 3 (Document Loaders)에서 지원 예정"
        )
    if not source_data:
        raise ValueError("source_data is required for text type")

    try:
        from structverify.core.pipeline import VerificationPipeline
    except ImportError as e:
        logger.error("structverify library not installed: %s", e)
        raise RuntimeError("structverify 라이브러리가 설치되어 있지 않습니다.")

    import os
    if settings.llm.api_key:
        os.environ.setdefault("CLOVASTUDIO_API_KEY", settings.llm.api_key)
    if settings.kosis.api_key:
        os.environ.setdefault("KOSIS_API_KEY", settings.kosis.api_key)

    pipeline = VerificationPipeline()
    report = await pipeline.run(source_data, source_type)

    return _build_response(report)


# ── 응답 정제 ──────────────────────────────────────────────────
KNOWN_VERDICTS = {"match", "mismatch", "partial", "unverifiable"}
VERDICT_ALIASES = {
    "supported": "match", "true": "match", "verified": "match",
    "consistent": "match", "correct": "match",
    "refuted": "mismatch", "false": "mismatch",
    "contradicted": "mismatch", "inconsistent": "mismatch",
    "incorrect": "mismatch", "wrong": "mismatch",
    "partially_supported": "partial", "partly_match": "partial",
    "mixed": "partial",
    "unknown": "unverifiable", "insufficient": "unverifiable",
    "no_evidence": "unverifiable", "not_enough_info": "unverifiable",
    "nei": "unverifiable", "none": "unverifiable",
}


def _normalize_verdict(val: Any) -> str | None:
    """단일 값을 표준 verdict로 정규화. 매칭 안 되면 None."""
    if not isinstance(val, str):
        return None
    s = val.lower().strip()
    if not s or s in ("none", "null"):
        return None
    if s in KNOWN_VERDICTS:
        return s
    if s in VERDICT_ALIASES:
        return VERDICT_ALIASES[s]
    # "Verdict.MATCH" 같은 enum repr 처리
    if "." in s:
        tail = s.rsplit(".", 1)[1]
        if tail in KNOWN_VERDICTS:
            return tail
        if tail in VERDICT_ALIASES:
            return VERDICT_ALIASES[tail]
    return None


def _build_response(report: Any) -> dict[str, Any]:
    """
    라이브러리 report → 프론트 호환 dict.

    라이브러리 구조 (확정):
        report.claims  = [{claim_id, sent_id, claim_text, schema, ...}, ...]
        report.results = [{claim_id, verdict, evidence, explanation, ...}, ...]

    둘을 claim_id로 inner join 해서 통합된 claim 객체 만듦.
    """
    full = _safe_serialize(report, set())
    if not isinstance(full, dict):
        full = {}

    claims_raw = full.get("claims") or []
    results_raw = full.get("results") or []
    if not isinstance(claims_raw, list):
        claims_raw = []
    if not isinstance(results_raw, list):
        results_raw = []

    # ── 진단 로그: claim/result 양쪽 키 구조 확인 ──
    if claims_raw and isinstance(claims_raw[0], dict):
        logger.warning("[DEBUG] Claim keys: %s", sorted(claims_raw[0].keys()))
    if results_raw and isinstance(results_raw[0], dict):
        logger.warning("[DEBUG] Result keys: %s", sorted(results_raw[0].keys()))
        # 첫 result의 verdict-like 값들 확인
        sample = results_raw[0]
        logger.warning(
            "[DEBUG] First result peek: verdict=%r, evidence=%s, explanation=%s",
            sample.get("verdict"),
            type(sample.get("evidence")).__name__,
            type(sample.get("explanation")).__name__,
        )

    # claim_id → result 매핑
    result_by_claim_id: dict[str, dict] = {}
    for r in results_raw:
        if not isinstance(r, dict):
            continue
        cid = r.get("claim_id")
        if cid:
            result_by_claim_id[str(cid)] = r

    # claim + result 합치기 + verdict 정규화
    distribution = {"match": 0, "mismatch": 0, "partial": 0, "unverifiable": 0}
    merged_claims = []
    for c in claims_raw:
        if not isinstance(c, dict):
            continue

        cid = c.get("claim_id")
        r = result_by_claim_id.get(str(cid)) if cid else None

        # 합치기 (result 필드가 우선)
        merged = dict(c)
        if isinstance(r, dict):
            merged.update(r)

        # verdict 정규화 — multi-key fallback
        v_norm = None
        for key in ("verdict", "decision", "status", "result", "outcome"):
            v_norm = _normalize_verdict(merged.get(key))
            if v_norm:
                break
        v_norm = v_norm or "unverifiable"
        merged["verdict"] = v_norm
        distribution[v_norm] += 1
        merged_claims.append(merged)

    # domain, anchor_year 안전 추출
    domain = full.get("domain")
    if not isinstance(domain, (str, type(None))):
        domain = str(domain) if domain else None

    anchor_year = full.get("anchor_year")
    if not isinstance(anchor_year, (int, type(None))):
        try:
            anchor_year = int(anchor_year) if anchor_year else None
        except (ValueError, TypeError):
            anchor_year = None

    return {
        "domain": domain,
        "anchor_year": anchor_year,
        "claims": merged_claims,
        "verdict_distribution": distribution,
    }


# ── 직렬화 ────────────────────────────────────────────────────────
def _safe_serialize(obj: Any, seen: set[int]) -> Any:
    """순환 끊으며 JSON-safe로 변환."""
    from datetime import datetime, date
    from decimal import Decimal
    from enum import Enum
    from uuid import UUID

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return obj.hex()

    obj_id = id(obj)
    if obj_id in seen:
        return None

    seen.add(obj_id)
    try:
        if isinstance(obj, dict):
            return {str(k): _safe_serialize(v, seen) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set, frozenset)):
            return [_safe_serialize(v, seen) for v in obj]

        if hasattr(obj, "__dict__"):
            return {
                k: _safe_serialize(v, seen)
                for k, v in vars(obj).items()
                if not k.startswith("_")
            }
    finally:
        seen.discard(obj_id)

    return str(obj)