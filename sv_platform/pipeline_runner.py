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
from structverify.utils.logger import get_logger
import re
from datetime import datetime, timezone
from typing import Any
from uuid import UUID
import logging
from sv_platform.config import settings

logger = get_logger(__name__)


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
    _check_input(source_type, source_data, source_uri)
    _inject_env()

    from structverify.core.pipeline import VerificationPipeline
    # 동기 진입점은 API job_id가 없음 — config의 scope이 "job_id"면
    # 라이브러리가 fresh UUID 발급.
    pipeline = VerificationPipeline()
    pipeline_source = _resolve_pipeline_source(source_type, source_data, source_uri)
    report = await pipeline.run(pipeline_source, source_type)
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
        _check_input(source_type, source_data, source_uri)
        _inject_env()

        # 4) 라이브러리 실행
        # [2026-05-21] workspace scope="job_id" 모드에서 사용할 API job_id 주입.
        # scope="doc_hash"면 라이브러리가 무시. config 로드 → external_job_id 박고
        # pipeline에 전달.
        from structverify.core.pipeline import VerificationPipeline
        from structverify.core.config_loader import load_config
        _cfg = load_config()
        _cfg.setdefault("agent", {}).setdefault("workspace", {})["external_job_id"] = str(job_id)
        pipeline = VerificationPipeline(config=_cfg)
        pipeline_source = _resolve_pipeline_source(source_type, source_data, source_uri)

        # [2026-05-21] URL 입력 사전 추출 — 본문을 *pipeline 시작 전*에 추출해
        # Job.source_data에 즉시 저장. 이유:
        #   1) 프론트가 /v1/jobs/{id} 폴링할 때 source_data 매칭으로 workspace
        #      디렉토리(scope=doc_hash 모드)를 찾아 partial 진행상황을 노출.
        #      source_data=None이면 디렉토리 매칭 실패 → 로딩 화면이 빈 채로 정지.
        #   2) 추출본을 미리 보여줘 사용자에게 "검증 진행 중" 시각화.
        # text 입력은 이미 source_data가 채워져 있어 영향 X.
        pre_extracted_text: str | None = None
        if source_type == "url" and source_uri:
            # [2026-05-21] URL 추출 *시작 전* 상태 업데이트 — 프론트가 폴링 시
            # "본문 추출 중" 상태를 즉시 보여주도록. trafilatura/LLM scraper로
            # 30초 이상 걸리는 동안 UI가 빈 채로 멈춰 보이는 문제 차단.
            try:
                async with _session_factory() as db_extracting:
                    job_extracting = await db_extracting.get(Job, job_id)
                    if job_extracting is not None:
                        job_extracting.current_step = "URL 본문 추출 중"
                        if job_extracting.progress < 8:
                            job_extracting.progress = 8
                        await db_extracting.commit()
            except Exception as _e:
                logger.debug(f"URL 추출 시작 상태 업데이트 실패 (무시): {_e}")

            try:
                from structverify.preprocessing.extractor import extract_text
                from structverify.core.schemas import SourceType as _SourceType
                pre_extracted_text = await extract_text(source_uri, _SourceType.URL)
                if pre_extracted_text:
                    # 즉시 DB 반영 — 별도 세션 사용 (현재 컨텍스트에서 락 풀린 상태)
                    async with _session_factory() as db_pre:
                        job_pre = await db_pre.get(Job, job_id)
                        if job_pre is not None and not job_pre.source_data:
                            job_pre.source_data = pre_extracted_text
                            job_pre.current_step = "본문 추출 완료"
                            if job_pre.progress < 12:
                                job_pre.progress = 12
                            await db_pre.commit()
                    logger.info(
                        f"URL 사전 추출 완료: {len(pre_extracted_text)}자 — "
                        f"Job.source_data 즉시 반영 (실시간 partial 매칭 가능)"
                    )
            except Exception as _e:
                logger.warning(f"URL 사전 추출 실패 (무시, pipeline 재시도): {_e}")
                pre_extracted_text = None

        # pipeline.run()에 사전 추출 텍스트 전달 — 중복 호출 회피
        report = await pipeline.run(
            pipeline_source,
            source_type,
            source_text=pre_extracted_text,
        )

        # [2026-05-21] URL/PDF 입력: pipeline이 추출한 raw_text를 source_data로
        # 승격. 프론트 "원문" 패널이 Job.source_data만 보고 렌더하기 때문.
        # text 입력은 source_data가 이미 채워져 있어 영향 X.
        extracted_text: str | None = pre_extracted_text
        if extracted_text is None:
            try:
                doc = getattr(report, "document", None)
                extracted_text = getattr(doc, "raw_text", None) if doc else None
            except Exception:
                extracted_text = None
        effective_source_text = source_data or extracted_text

        # job_id + source_text를 함께 넘겨, agent_workspace에서 claim별 plan/trace
        # 읽어 합침. source_text는 라이브러리가 자체 doc_id로 워크스페이스를
        # 만든 경우의 fallback 매칭용 (sv_platform job_id로 정확 매칭 실패 시).
        # URL 입력도 이제 추출된 본문으로 fallback 매칭 가능.
        result = _build_response(report, job_id=str(job_id), source_text=effective_source_text)

        # 5) job → completed (+ URL/PDF면 추출본을 source_data로 저장)
        async with _session_factory() as db:
            job = await db.get(Job, job_id)
            job.status = "completed"
            job.result = result
            job.progress = 100
            job.current_step = "완료"
            job.completed_at = datetime.now(timezone.utc)
            # URL 입력은 원래 source_data=None인데 추출된 본문이 있으면 박아서
            # 프론트가 "원문" 패널 렌더 가능하게 함. text 입력은 그대로.
            if extracted_text and not job.source_data:
                job.source_data = extracted_text
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
def _check_input(
    source_type: str,
    source_data: str | None,
    source_uri: str | None = None,
) -> None:
    if source_type == "text":
        if not source_data:
            raise ValueError("source_data is required for text type")
    elif source_type == "url":
        if not source_uri:
            raise ValueError("source_uri (url) is required for url type")
    else:
        # pdf/docx는 multipart 업로드 라우트(Phase 3)에서 처리 예정
        raise NotImplementedError(
            f"source_type='{source_type}'은 Phase 3에서 지원 예정 (파일 업로드)"
        )


def _resolve_pipeline_source(
    source_type: str,
    source_data: str | None,
    source_uri: str | None,
) -> str:
    """structverify VerificationPipeline.run(source=...) 의 첫 인자 결정.

    - text : 본문 문자열을 그대로
    - url  : URL 문자열 (라이브러리 내 extract_text가 trafilatura/LLM scraper로 추출)
    """
    if source_type == "url":
        return source_uri or ""
    return source_data or ""


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


def _build_response(
    report: Any,
    job_id: str | None = None,
    source_text: str | None = None,
) -> dict[str, Any]:
    """라이브러리 report → 프론트 호환 dict. claims+results inner join + 경량화.

    [확장] job_id 주어지면 agent_workspace에서 claim별 plan/trace 읽어 합침 (A-1).
    source_text 같이 주면 라이브러리 doc_id 워크스페이스를 fallback으로 매칭.
    프론트는 ClaimResult에 `plan`, `trace` 필드 추가로 받음.
    """
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

    # [A-1] 모든 claim_id를 모아 workspace에서 plan/trace 일괄 읽기
    workspace_by_cid: dict[str, dict] = {}
    if job_id:
        try:
            from sv_platform.loaders.workspace_reader import (
                read_job_workspace_for_claims,
            )
            all_cids = [
                str(c.get("claim_id"))
                for c in claims_raw
                if isinstance(c, dict) and c.get("claim_id")
            ]
            workspace_by_cid = read_job_workspace_for_claims(
                job_id, all_cids, source_text=source_text,
            )
        except Exception as e:
            logger.debug(f"[_build_response] workspace 읽기 실패: {e}")

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
        # [2026-05-21 P7] supporting_evidence도 같은 방식으로 경량화.
        # derived claim의 prev 시점 fetch 등 보조 데이터 리스트.
        _supp_raw = merged.get("supporting_evidence") or []
        if isinstance(_supp_raw, list):
            merged["supporting_evidence"] = [
                s for s in (_summarize_evidence(x) for x in _supp_raw)
                if s is not None
            ]
        else:
            merged["supporting_evidence"] = []

        # ── [A-1] plan + trace 합치기 ──
        ws_info = workspace_by_cid.get(str(cid), {}) if cid else {}
        if ws_info.get("plan"):
            merged["plan"] = ws_info["plan"]
        if ws_info.get("trace"):
            merged["trace"] = ws_info["trace"]

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