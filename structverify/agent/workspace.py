"""
structverify.agent.workspace — Agent 작업 공간 추상화.

각 검증 job마다 workspace 디렉토리가 생성되며, agent는 이 공간을
*파일 시스템처럼* 읽고 쓰면서 작업한다. memory.md, plan.json,
observation log 등을 *멀티턴 동안 누적*.

디렉토리 구조:
    workspace/job_{job_id}/
    ├─ meta.json                       # job 메타
    ├─ source.txt                      # 원문 기사 (raw)
    ├─ claims/
    │   └─ {claim_id}/
    │       ├─ claim.json              # ClaimSchema (입력)
    │       ├─ plan.json               # Plan Agent 출력
    │       ├─ memory.md               # 평문 메모리 (LLM이 read/write)
    │       ├─ log.jsonl               # 모든 action+observation
    │       ├─ observations/
    │       │   ├─ obs_001_catalog.json
    │       │   └─ obs_002_kosis.json
    │       ├─ data_points.json        # 모은 데이터 점들
    │       └─ verdict.json            # 최종 판정
    └─ summary.json

백엔드 추상화 (config.agent.workspace.backend):
    - "local": LocalWorkspaceBackend (현재 구현)
    - "minio": MinIOWorkspaceBackend (Phase F에서 구현, 지금은 NotImplementedError)
    - "s3":   같음
"""
from __future__ import annotations
import json
from structverify.utils.logger import get_logger
import shutil
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

logger = get_logger(__name__)


# ── [수정 v6.22] verified_facts 캐시 매칭 헬퍼 ────────────────────────
# [추가 이유] 기존 lookup_verified_fact의 포함관계 매칭이 너무 느슨해서,
#   '출생아 수 증가율'(%) claim이 '출생아 수'(명) 값 230028을 그대로
#   재사용하던 버그가 있었음 (단위가 % vs 명으로 다른데도 적중).
#   → 파생 지표(증가율/차이 등)를 base 지표와 분리하고, unit이 호환될
#     때만 캐시 적중을 허용하도록 아래 헬퍼를 추가.
# 도메인 무관 — 보편적 파생 어휘 + 단위 정규화만 사용.
_DERIVED_SUFFIXES = ("증가율", "감소율", "증감률", "변화율", "상승률",
                     "하락률", "증감", "차이", "증가폭", "변화량")


def _strip_derived_suffix(indicator: str) -> str:
    """indicator에서 파생 접미사를 떼어 base 지표명을 반환.
    '출생아 수 증가율' → '출생아 수' / '합계출산율 차이' → '합계출산율'
    """
    s = str(indicator or "").strip()
    for suf in _DERIVED_SUFFIXES:
        if s.endswith(suf):
            return s[: -len(suf)].strip()
    return s


def _norm_unit(unit: str) -> str:
    """단위 문자열 정규화 — 캐시 매칭 시 % ↔ 명 혼동 차단용.
    퍼센트 계열 → 'percent', 그 외는 공백 제거한 원문.
    """
    u = str(unit or "").strip().lower()
    if not u:
        return ""
    if u in ("%", "％", "percent", "퍼센트", "프로", "pp", "%p", "퍼센트포인트"):
        return "percent"
    return u.replace(" ", "")


# ── 백엔드 추상 인터페이스 ─────────────────────────────────────────

class WorkspaceBackend(ABC):
    """Workspace 저장소의 추상 인터페이스 (key-value text 저장)."""

    @abstractmethod
    def read_text(self, key: str) -> str:
        """key의 내용을 읽음. 없으면 FileNotFoundError."""

    @abstractmethod
    def write_text(self, key: str, content: str) -> None:
        """key에 내용 쓰기 (덮어쓰기). 상위 경로 자동 생성."""

    @abstractmethod
    def append_text(self, key: str, content: str) -> None:
        """key에 내용 추가 (append). 없으면 생성."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """key 존재 여부."""

    @abstractmethod
    def list_keys(self, prefix: str) -> list[str]:
        """prefix 아래 모든 key (재귀)."""

    @abstractmethod
    def delete_prefix(self, prefix: str) -> None:
        """prefix 아래 모든 항목 삭제 (cleanup용)."""


# ── Local 백엔드 (기본 구현) ─────────────────────────────────────

class LocalWorkspaceBackend(WorkspaceBackend):
    """파일 시스템 백엔드. key는 슬래시 구분 경로."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        logger.debug(f"[workspace] LocalWorkspaceBackend root={self.root}")

    def _path(self, key: str) -> Path:
        # 경로 traversal 방지 — '..' 같은 거 차단
        if ".." in Path(key).parts:
            raise ValueError(f"Invalid key (path traversal): {key!r}")
        return self.root / key

    def read_text(self, key: str) -> str:
        return self._path(key).read_text(encoding="utf-8")

    def write_text(self, key: str, content: str) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def append_text(self, key: str, content: str) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(content)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def list_keys(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        if not base.exists():
            return []
        return [
            str(p.relative_to(self.root)).replace("\\", "/")
            for p in base.rglob("*")
            if p.is_file()
        ]

    def delete_prefix(self, prefix: str) -> None:
        p = self._path(prefix)
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()


# ── MinIO/S3 백엔드 (자리만, 미구현) ───────────────────────────────

class MinIOWorkspaceBackend(WorkspaceBackend):
    """MinIO/S3 백엔드 — Phase F에서 구현 예정.

    config.storage.* 인프라(이미 yaml에 있음)와 통합 가능.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "MinIO 백엔드는 아직 구현되지 않았습니다. "
            "config: agent.workspace.backend='local'을 사용하세요. "
            "Phase F (배포)에서 구현 예정."
        )

    def read_text(self, key): raise NotImplementedError
    def write_text(self, key, content): raise NotImplementedError
    def append_text(self, key, content): raise NotImplementedError
    def exists(self, key): raise NotImplementedError
    def list_keys(self, prefix): raise NotImplementedError
    def delete_prefix(self, prefix): raise NotImplementedError


# ── Workspace 메인 클래스 (도메인 API) ───────────────────────────

class Workspace:
    """
    Workspace API — 백엔드 위에 *agent용 도메인 메서드* 제공.

    Phase A에서는 *저장/조회 기능*만 제공. 실제 멀티턴 loop은 Phase D에서
    이 클래스 사용. Phase B에서는 tool 인터페이스가 이 클래스를 통해
    workspace 파일을 직접 read/write.

    Usage:
        ws = build_workspace(job_id, config={"backend": "local", "local_path": "./ws"})
        ws.initialize(source_text="기사 원문", meta={"datasources": ["kosis"]})
        ws.create_claim_dir(claim_id, claim_data)
        ws.append_memory(claim_id, "## Iteration 1\\n...")
        memory = ws.read_memory(claim_id)
    """

    def __init__(self, job_id: str | UUID, backend: WorkspaceBackend):
        self.job_id = str(job_id)
        self.backend = backend
        self._prefix = f"job_{self.job_id}"

    # ── 키 생성 헬퍼 ────────────────────────────────────────────
    def _meta_key(self) -> str:
        return f"{self._prefix}/meta.json"

    def _source_key(self) -> str:
        return f"{self._prefix}/source.txt"

    def _summary_key(self) -> str:
        return f"{self._prefix}/summary.json"

    def _claim_dir(self, claim_id: str | UUID) -> str:
        return f"{self._prefix}/claims/{claim_id}"

    def _claim_file(self, claim_id: str | UUID, name: str) -> str:
        return f"{self._claim_dir(claim_id)}/{name}"
    
    def is_initialized(self) -> bool:
        return self.backend.exists(self._meta_key())

    # ── 초기화 ───────────────────────────────────────────────
    def initialize(self, source_text: str, meta: dict | None = None) -> None:
        """Job 시작 시 호출. source + meta 저장."""
        meta = dict(meta) if meta else {}
        meta.setdefault("job_id", self.job_id)
        meta.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        self.backend.write_text(
            self._meta_key(),
            json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        )
        self.backend.write_text(self._source_key(), source_text)
        logger.info(f"[workspace] initialized: job_id={self.job_id}")

    def create_claim_dir(self, claim_id: str | UUID, claim_data: dict) -> None:
        """claim 작업 디렉토리 생성 + 빈 memory 초기화."""
        cid = str(claim_id)
        self.backend.write_text(
            self._claim_file(cid, "claim.json"),
            json.dumps(claim_data, ensure_ascii=False, indent=2, default=str),
        )
        # memory.md 초기화 (헤더만)
        self.backend.write_text(
            self._claim_file(cid, "memory.md"),
            f"# Memory for claim {cid}\n\n"
            f"Created at: {datetime.now(timezone.utc).isoformat()}\n\n",
        )
        logger.info(f"[workspace] claim dir created: {cid}")

    # ── Source (원문 기사) ───────────────────────────────────
    def read_source(self) -> str:
        """원문 기사 전체."""
        return self.backend.read_text(self._source_key())

    def read_source_span(self, start: int = 0, end: int | None = None) -> str:
        """원문의 일부 (글자 인덱스). agent가 부분만 읽을 때."""
        text = self.read_source()
        return text[start:end] if end is not None else text[start:]

    # ── Meta ────────────────────────────────────────────────
    def read_meta(self) -> dict:
        return json.loads(self.backend.read_text(self._meta_key()))

    # ── Claim ───────────────────────────────────────────────
    def read_claim(self, claim_id: str | UUID) -> dict:
        return json.loads(self.backend.read_text(self._claim_file(str(claim_id), "claim.json")))

    def list_claims(self) -> list[str]:
        """모든 claim_id 목록."""
        keys = self.backend.list_keys(f"{self._prefix}/claims")
        cids = set()
        for k in keys:
            # job_xxx/claims/{cid}/...
            parts = k.split("/")
            if len(parts) >= 3 and parts[1] == "claims":
                cids.add(parts[2])
        return sorted(cids)

    # ── Plan ────────────────────────────────────────────────
    def write_plan(self, claim_id: str | UUID, plan_data: dict) -> None:
        self.backend.write_text(
            self._claim_file(str(claim_id), "plan.json"),
            json.dumps(plan_data, ensure_ascii=False, indent=2, default=str),
        )

    def read_plan(self, claim_id: str | UUID) -> dict | None:
        key = self._claim_file(str(claim_id), "plan.json")
        if not self.backend.exists(key):
            return None
        return json.loads(self.backend.read_text(key))

    # ── Memory (markdown, append-only) ───────────────────────
    def append_memory(self, claim_id: str | UUID, text: str) -> None:
        """memory.md에 텍스트 추가. 끝에 줄바꿈 추가됨."""
        if not text.endswith("\n"):
            text += "\n"
        self.backend.append_text(self._claim_file(str(claim_id), "memory.md"), text)

    def read_memory(self, claim_id: str | UUID) -> str:
        return self.backend.read_text(self._claim_file(str(claim_id), "memory.md"))

    # ── [v6.21] Verified Facts (job 레벨 공유 메모리) ────────────
    # claim 디렉토리 밖, job 레벨에 둬서 모든 claim이 공유한다.
    # 한 claim에서 검증한 수치를 다음 claim이 재검색 없이 재사용.
    # 예: "올해 출생아 수 20,717명" 검증 후, "작년 대비 8.7% 증가"
    #     claim은 올해값을 catalog_search 없이 즉시 가져온다.
    def _facts_key(self) -> str:
        return f"{self._prefix}/verified_facts.json"

    def _successful_stat_ids_key(self) -> str:
        return f"{self._prefix}/successful_stat_ids.json"

    def read_successful_stat_ids(self) -> list[str]:
        """[패치 A] job에서 fetch_evidence가 성공한 stat_table_id 목록.

        같은 KOSIS 표(예: DT_1B8000G)에 출생아 수, 합계출산율, 혼인 건수가
        모두 있는데 catalog는 검색어별로 다른 표를 top으로 줘서 같은 표 다른
        row를 못 받는 케이스 대응. 한 claim이 표에서 성공하면 그 stat_id를
        저장해 두고, 다음 claim의 fetch fallback 후보 맨 앞에 prepend한다.
        """
        key = self._successful_stat_ids_key()
        if not self.backend.exists(key):
            return []
        try:
            data = json.loads(self.backend.read_text(key))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def append_successful_stat_id(self, stat_id: str) -> None:
        """fetch_evidence success 시 stat_id를 job 공유 list에 추가 (중복 X)."""
        sid = (stat_id or "").strip()
        if not sid:
            return
        ids = self.read_successful_stat_ids()
        if sid in ids:
            return
        ids.append(sid)
        self.backend.write_text(
            self._successful_stat_ids_key(),
            json.dumps(ids, ensure_ascii=False, indent=2),
        )
        logger.info(
            f"[workspace] successful_stat_id 저장: {sid!r} "
            f"(누적 {len(ids)}개) — 다음 claim fetch fallback 1순위로 사용"
        )

    def read_verified_facts(self) -> list[dict]:
        """job에서 지금까지 검증된 사실 목록.

        각 항목: {indicator, time_period, value, unit, source, claim_id, verdict}
        """
        key = self._facts_key()
        if not self.backend.exists(key):
            return []
        try:
            return json.loads(self.backend.read_text(key))
        except Exception:
            return []

    def append_verified_fact(self, fact: dict) -> None:
        """검증 완료된 수치를 job 공유 저장소에 추가.

        fact: {indicator, time_period, value, unit, source, claim_id, verdict}
        동일 (indicator, time_period) 항목이 이미 있으면 덮어쓰지 않고
        그대로 둔다 (최초 검증 결과 우선 — 일관성).
        """
        if not fact or fact.get("value") is None:
            return
        facts = self.read_verified_facts()
        ind = str(fact.get("indicator", "") or "").strip()
        tp = str(fact.get("time_period", "") or "").strip()
        for f in facts:
            if (str(f.get("indicator", "") or "").strip() == ind
                    and str(f.get("time_period", "") or "").strip() == tp):
                return  # 이미 있음 — 중복 저장 안 함
        fact = dict(fact)
        fact.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
        facts.append(fact)
        self.backend.write_text(
            self._facts_key(),
            json.dumps(facts, ensure_ascii=False, indent=2, default=str),
        )
        logger.info(
            f"[workspace] verified_fact 저장: "
            f"indicator={ind!r} time={tp!r} value={fact.get('value')}"
        )

    def lookup_verified_fact(
        self, indicator: str, time_period: str, unit_hint: str | None = None
    ) -> dict | None:
        """(indicator, time_period)로 검증된 사실 조회. 없으면 None.

        [수정 v6.22] 매칭 규칙을 엄격하게 — 잘못된 캐시 재사용 방지.
        [BEFORE 버그] indicator 포함관계('A' in 'A 증가율')만으로 매칭 →
          '출생아 수 증가율'(%) claim이 '출생아 수'(명) 230028을 재사용.
        [AFTER 수정]
          1) indicator + time_period 정확 일치 (+ unit 호환)
          2) base indicator(파생 접미사 제거) 일치 + unit 호환
        '증가율'·'차이' 같은 파생 지표는 원지표와 unit이 다르므로
        unit_hint 가드가 자동으로 걸러낸다.

        unit_hint가 주어지면, 저장된 fact의 unit과 호환되지 않는 항목은
        매칭에서 제외 (% ↔ 명 혼동 차단).
        """
        ind = str(indicator or "").strip()
        tp = str(time_period or "").strip()
        if not ind or not tp:
            return None
        facts = self.read_verified_facts()

        def _unit_ok(fact_unit: str) -> bool:
            """unit_hint와 fact의 unit이 호환되는지. hint 없으면 통과."""
            if not unit_hint:
                return True
            fu = _norm_unit(fact_unit)
            hu = _norm_unit(unit_hint)
            if not fu or not hu:
                return True  # 한쪽이라도 비었으면 판별 불가 → 통과
            return fu == hu

        # 1차: indicator + time_period 정확 일치 (+ unit 호환)
        for f in facts:
            if (str(f.get("indicator", "") or "").strip() == ind
                    and str(f.get("time_period", "") or "").strip() == tp):
                if _unit_ok(str(f.get("unit", "") or "")):
                    return f
                logger.info(
                    f"[workspace] verified_fact unit 불일치로 캐시 거부: "
                    f"{ind} {tp} (요청 unit={unit_hint}, "
                    f"저장 unit={f.get('unit')})"
                )
                return None

        # 2차: base indicator(파생 접미사 제거) 일치 + unit 호환
        # '증가율'/'차이' 지표는 base가 같아도 unit이 다르므로 _unit_ok가
        # 걸러준다. 즉 안전한 경우(같은 단위·동일 base 지표)만 재사용.
        ind_base = _strip_derived_suffix(ind)
        for f in facts:
            if str(f.get("time_period", "") or "").strip() != tp:
                continue
            f_ind = str(f.get("indicator", "") or "").strip()
            f_base = _strip_derived_suffix(f_ind)
            if ind_base and f_base and ind_base == f_base:
                if _unit_ok(str(f.get("unit", "") or "")):
                    return f
        return None

    # ── Log (jsonl, append-only — 구조화) ────────────────────
    def append_log(self, claim_id: str | UUID, entry: dict) -> None:
        """Action+Observation 한 줄 추가 (JSON line)."""
        entry = dict(entry)
        entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        line = json.dumps(entry, ensure_ascii=False, default=str)
        self.backend.append_text(
            self._claim_file(str(claim_id), "log.jsonl"),
            line + "\n",
        )

    def read_log(self, claim_id: str | UUID) -> list[dict]:
        key = self._claim_file(str(claim_id), "log.jsonl")
        if not self.backend.exists(key):
            return []
        text = self.backend.read_text(key)
        return [json.loads(line) for line in text.strip().split("\n") if line.strip()]

    # ── Observation (개별 tool 호출 raw 결과) ──────────────────
    def write_observation(self, claim_id: str | UUID, name: str, data: dict) -> None:
        """tool 호출 결과를 별도 파일로 (디버깅용 raw)."""
        cid = str(claim_id)
        key = f"{self._claim_dir(cid)}/observations/{name}.json"
        self.backend.write_text(
            key,
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
        )

    def list_observations(self, claim_id: str | UUID) -> list[str]:
        cid = str(claim_id)
        prefix = f"{self._claim_dir(cid)}/observations"
        keys = self.backend.list_keys(prefix)
        return sorted([k.rsplit("/", 1)[-1] for k in keys])

    def read_observation(self, claim_id: str | UUID, name: str) -> dict | None:
        cid = str(claim_id)
        key = f"{self._claim_dir(cid)}/observations/{name}"
        if not self.backend.exists(key):
            return None
        try:
            return json.loads(self.backend.read_text(key))
        except (json.JSONDecodeError, OSError) as e:
            logger.debug(f"[workspace] observation 읽기 실패 {name}: {e}")
            return None

    # ── Data points (모은 수치들) ─────────────────────────────
    def write_data_points(self, claim_id: str | UUID, points: list[dict]) -> None:
        self.backend.write_text(
            self._claim_file(str(claim_id), "data_points.json"),
            json.dumps(points, ensure_ascii=False, indent=2, default=str),
        )

    def read_data_points(self, claim_id: str | UUID) -> list[dict]:
        key = self._claim_file(str(claim_id), "data_points.json")
        if not self.backend.exists(key):
            return []
        return json.loads(self.backend.read_text(key))

    # ── Verdict (최종 판정) ──────────────────────────────────
    def write_verdict(self, claim_id: str | UUID, verdict_data: dict) -> None:
        self.backend.write_text(
            self._claim_file(str(claim_id), "verdict.json"),
            json.dumps(verdict_data, ensure_ascii=False, indent=2, default=str),
        )

    def read_verdict(self, claim_id: str | UUID) -> dict | None:
        key = self._claim_file(str(claim_id), "verdict.json")
        if not self.backend.exists(key):
            return None
        return json.loads(self.backend.read_text(key))

    # ── Summary (모든 claim 종합) ─────────────────────────────
    def write_summary(self, summary_data: dict) -> None:
        self.backend.write_text(
            self._summary_key(),
            json.dumps(summary_data, ensure_ascii=False, indent=2, default=str),
        )

    # ── Cleanup ──────────────────────────────────────────────
    def cleanup(self) -> None:
        """이 job의 모든 파일 삭제 (config.agent.workspace.persist_after_job=false 시)."""
        self.backend.delete_prefix(self._prefix)
        logger.info(f"[workspace] cleaned up: job_id={self.job_id}")


# ── Factory ─────────────────────────────────────────────────────

def build_workspace(
    job_id: str | UUID,
    config: dict | None = None,
) -> Workspace:
    """
    Config에서 workspace 인스턴스 생성.

    config 예 (config/default.yaml의 agent.workspace 섹션):
        backend: "local"
        local_path: "./agent_workspace"

    Args:
        job_id: 검증 job 식별자.
        config: agent.workspace 섹션 dict. None이면 default.
    """
    config = config or {}
    backend_type = config.get("backend", "local")

    if backend_type == "local":
        path = config.get("local_path", "./agent_workspace")
        backend = LocalWorkspaceBackend(path)
    elif backend_type in ("minio", "s3"):
        backend = MinIOWorkspaceBackend(config)
    else:
        raise ValueError(
            f"Unknown workspace backend: {backend_type!r}. "
            f"지원: 'local' (현재), 'minio'/'s3' (Phase F 예정)"
        )

    return Workspace(job_id=job_id, backend=backend)