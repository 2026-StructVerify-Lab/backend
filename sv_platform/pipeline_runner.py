"""
sv_platform.pipeline_runner — `structverify` 라이브러리 호출 wrapper

[Phase 1 → Phase 2 이전]
- 동기 호출 `run_verification()` 유지 (백워드 호환)
- 새 비동기 background task `run_verification_background()` 추가:
  · 자체 DB 세션
  · structverify logger handler 가로채서 job.current_step / progress 업데이트
  · 라이브러리 로그 라인을 job.log_messages JSON 배열에 누적
  · 완료/실패 시 job 레코드 최종 업데이트

[설계 노트]
- BackgroundTask는 단일 uvicorn worker 안에서 asyncio task로 실행 → 메모리 공유.
- 진짜 분산 큐(ARQ/Redis)는 Phase 2 후반에 도입.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sv_platform.config import settings

logger = logging.getLogger(__name__)


# ── 단계별 진행률 매핑 ────────────────────────────────────────────
# structverify logger의 step 메시지를 캐치해서 progress 업데이트.
# 라이브러리 로그 패턴 (실제 출력 기반):
#   [Agent A] Step 3 classify_domain → ...
#   [Agent A] Step 4 detect_claims → ...
#   [Agent A] Step 5 induce_schemas → ...
#   [Agent A] Step 6 build_claim_graph → ...
#   [Agent A] Step 7 retrieve_evidence → ...
#   [Agent A] Step 8 verify_claim → ...
#   [Agent A] Step 9 generate_explanation → ...
#   [Agent A] 완료: claims=...
STEP_PROGRESS = {
    "classify_domain":              (15, "도메인 분류"),
    "detect_claims":                (25, "검증 가능 주장 탐지"),
    "build_document_temporal_graph": (30, "시간 그래프 구축"),
    "induce_schemas":               (40, "스키마 유도"),
    "build_claim_graph":            (50, "Claim 그래프 빌드"),
    "retrieve_evidence":            (65, "공식 통계 조회"),
    "verify_claim":                 (80, "수치 검증"),
    "generate_explanation":         (90, "근거 설명 생성"),
}

STEP_PATTERN = re.compile(
    r"\[Agent\s+A\]\s+Step\s+[\d.]+\s+(\w+)",
    re.IGNORECASE,
)
COMPLETION_PATTERN = re.compile(r"\[Agent\s+A\]\s+완료", re.IGNORECASE)


# ── 동기 진입점 (Phase 1 호환) ────────────────────────────────────
async def run_verification(
    source_type: str,
    source_data: str | None = None,
    source_uri: str | None = None,
    datasources: list[str] | None = None,
) -> dict[str, Any]:
    """동기 실행 — verify.py가 호출. 결과 dict 반환."""
    _check_input(source_type, source_data)
    _inject_env()

    from structverify.core.pipeline import VerificationPipeline
    pipeline = VerificationPipeline()
    report = await pipeline.run(source_data, source_type)
    return _build_response(report)


# ── 비동기 백그라운드 진입점 (Phase 2) ──────────────────────────────
async def run_verification_background(
    job_id: UUID,
    source_type: str,
    source_data: str | None,
    source_uri: str | None,
    datasources: list[str] | None,
) -> None:
    """
    백그라운드 실행 — verify.py가 fire-and-forget으로 호출.

    1) 자체 DB 세션 열기
    2) job.status='running', started_at 기록
    3) structverify logger에 progress handler 붙이고 라이브러리 실행
    4) 결과 또는 에러로 job 최종 업데이트
    """
    # 지연 import (순환 의존 방지)
    from sv_platform.db import _session_factory
    from sv_platform.models.job import Job

    if _session_factory is None:
        logger.error("Session factory not initialized")
        return

    progress_handler: JobProgressLogHandler | None = None
    sv_logger: logging.Logger | None = None

    try:
        # 1) job 상태 → running
        async with _session_factory() as db:
            job = await db.get(Job, job_id)
            if job is None:
                logger.error(f"Background: job {job_id} not found")
                return
            job.status = "running"
            job.started_at = datetime.now(timezone.utc)
            job.current_step = "초기화"
            job.progress = 5
            await db.commit()

        # 2) logger handler 부착
        progress_handler = JobProgressLogHandler(job_id, _session_factory)
        sv_logger = logging.getLogger("structverify")
        sv_logger.addHandler(progress_handler)

        # 3) 입력 검증 + 환경 변수
        _check_input(source_type, source_data)
        _inject_env()

        # 4) 라이브러리 실행
        from structverify.core.pipeline import VerificationPipeline
        pipeline = VerificationPipeline()
        report = await pipeline.run(source_data, source_type)
        result = _build_response(report)

        # 5) job → completed
        async with _session_factory() as db:
            job = await db.get(Job, job_id)
            job.status = "completed"
            job.result = result
            job.progress = 100
            job.current_step = "완료"
            job.completed_at = datetime.now(timezone.utc)
            await db.commit()
        logger.info(f"Job {job_id} completed")

    except NotImplementedError as e:
        await _mark_failed(_session_factory, job_id, str(e))
    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        await _mark_failed(_session_factory, job_id, str(e))
    finally:
        # logger handler 분리
        if progress_handler and sv_logger:
            sv_logger.removeHandler(progress_handler)


async def _mark_failed(session_factory, job_id: UUID, error: str) -> None:
    from sv_platform.models.job import Job
    try:
        async with session_factory() as db:
            job = await db.get(Job, job_id)
            if job:
                job.status = "failed"
                job.error = error
                job.completed_at = datetime.now(timezone.utc)
                await db.commit()
    except Exception:
        logger.exception("Failed to mark job as failed")


# ── Logger Handler — job.progress / job.current_step 자동 업데이트 ──
class JobProgressLogHandler(logging.Handler):
    """
    structverify 로거를 가로채서 [Agent A] Step 메시지를 보면
    DB의 job.progress / job.current_step / job.log_messages 업데이트.

    log_messages는 JSON 컬럼이 *아닌* job.current_step만 갱신 (스키마 보호).
    추후 별도 `job_logs` 테이블 만들 때 본격 누적.
    """

    def __init__(self, job_id: UUID, session_factory):
        super().__init__()
        self.job_id = job_id
        self.session_factory = session_factory
        self._lock = asyncio.Lock()
        self._last_progress = 5

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()

        # 완료 메시지
        if COMPLETION_PATTERN.search(msg):
            self._schedule_update(95, "결과 정리")
            return

        # Step 메시지
        m = STEP_PATTERN.search(msg)
        if not m:
            return
        step_name = m.group(1)
        if step_name not in STEP_PROGRESS:
            return
        progress, label = STEP_PROGRESS[step_name]
        # 후퇴 방지
        if progress <= self._last_progress:
            return
        self._last_progress = progress
        self._schedule_update(progress, label)

    def _schedule_update(self, progress: int, label: str) -> None:
        """async DB 업데이트를 별도 task로 fire-and-forget."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._update(progress, label))
        except RuntimeError:
            pass  # 이벤트 루프 없음 (사실상 발생 안 함)

    async def _update(self, progress: int, label: str) -> None:
        from sv_platform.models.job import Job
        try:
            async with self.session_factory() as db:
                job = await db.get(Job, self.job_id)
                if job and job.progress < progress:
                    job.progress = progress
                    job.current_step = label
                    await db.commit()
        except Exception as e:
            # 로깅 안 함 — 무한 루프 위험
            pass


# ── 유틸 ─────────────────────────────────────────────────────────────
def _check_input(source_type: str, source_data: str | None) -> None:
    if source_type != "text":
        raise NotImplementedError(
            f"source_type='{source_type}'은 Phase 3에서 지원 예정"
        )
    if not source_data:
        raise ValueError("source_data is required for text type")


def _inject_env() -> None:
    import os
    if settings.llm.api_key:
        os.environ.setdefault("CLOVASTUDIO_API_KEY", settings.llm.api_key)
    if settings.kosis.api_key:
        os.environ.setdefault("KOSIS_API_KEY", settings.kosis.api_key)


# ── 응답 정제 ─────────────────────────────────────────────────────────
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
    if not isinstance(val, str):
        return None
    s = val.lower().strip()
    if not s or s in ("none", "null"):
        return None
    if s in KNOWN_VERDICTS:
        return s
    if s in VERDICT_ALIASES:
        return VERDICT_ALIASES[s]
    if "." in s:
        tail = s.rsplit(".", 1)[1]
        if tail in KNOWN_VERDICTS:
            return tail
        if tail in VERDICT_ALIASES:
            return VERDICT_ALIASES[tail]
    return None


def _build_response(report: Any) -> dict[str, Any]:
    """라이브러리 report → 프론트 호환 dict. claims+results inner join + 경량화."""
    full = _safe_serialize(report, set())
    if not isinstance(full, dict):
        full = {}

    claims_raw = full.get("claims") or []
    results_raw = full.get("results") or []
    if not isinstance(claims_raw, list): claims_raw = []
    if not isinstance(results_raw, list): results_raw = []

    # claim_id → result 매핑
    result_by_claim_id: dict[str, dict] = {}
    for r in results_raw:
        if not isinstance(r, dict): continue
        cid = r.get("claim_id")
        if cid:
            result_by_claim_id[str(cid)] = r

    # claim + result 합치기 + verdict 정규화 + evidence 경량화
    distribution = {"match": 0, "mismatch": 0, "partial": 0, "unverifiable": 0}
    merged_claims = []
    for c in claims_raw:
        if not isinstance(c, dict): continue
        cid = c.get("claim_id")
        r = result_by_claim_id.get(str(cid)) if cid else None
        merged = dict(c)
        if isinstance(r, dict):
            merged.update(r)

        # ── verdict 정규화 ──
        v_norm = None
        for key in ("verdict", "decision", "status", "result", "outcome"):
            v_norm = _normalize_verdict(merged.get(key))
            if v_norm: break
        v_norm = v_norm or "unverifiable"
        merged["verdict"] = v_norm
        distribution[v_norm] += 1

        # ── evidence 경량화 (9.4MB → 수십KB) ──
        # 라이브러리 evidence는 KOSIS 전체 raw 데이터를 dict로 들고 옴.
        # 프론트에서 필요한 건 *요약 정보*만:
        #   - stat_table_id, stat_name (출처)
        #   - official_value, unit (비교 대상 수치)
        #   - time_period (KOSIS 시점)
        merged["evidence"] = _summarize_evidence(merged.get("evidence"))

        merged_claims.append(merged)

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


def _summarize_evidence(evidence: Any) -> dict | None:
    """
    evidence 경량화 — 검증 결과 표시에 필요한 핵심 필드만.

    라이브러리는 KOSIS API 응답 전체를 evidence에 보관해서 한 claim당 수MB가 됨.
    프론트는 출처/공식수치/시점만 보여주면 충분.
    """
    if not isinstance(evidence, dict):
        return None

    # 라이브러리 evidence 키 (관찰됨):
    #   source_name, stat_table_id, official_value, unit, time_period
    #   raw_data (KOSIS 응답 전체 — *제외*)
    KEEP_KEYS = (
        "source_name", "stat_table_id", "stat_name",
        "official_value", "official_unit", "unit",
        "time_period", "period", "org_id", "org_name",
        "match_score", "relevance_score",
    )
    summary: dict[str, Any] = {}
    for k in KEEP_KEYS:
        if k in evidence and evidence[k] is not None:
            summary[k] = evidence[k]
    return summary or None



def _safe_serialize(obj: Any, seen: set[int]) -> Any:
    """순환 끊으며 JSON-safe로 변환."""
    from datetime import datetime as _dt, date as _d
    from decimal import Decimal
    from enum import Enum
    from uuid import UUID as _UUID

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (_dt, _d)):
        return obj.isoformat()
    if isinstance(obj, _UUID):
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