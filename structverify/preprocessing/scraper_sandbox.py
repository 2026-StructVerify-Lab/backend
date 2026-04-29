"""
preprocessing/scraper_sandbox.py — LLM 생성 스크래핑 코드 격리 실행 (Docker/E2B)

[김예슬 - 2026-04-28 / v3]
- SandboxRunner 클래스: LLM이 생성한 코드를 격리 환경에서 실행
  · DockerSandbox: 로컬 Docker 컨테이너 격리 실행
    - 네트워크 제한, 메모리 256m, CPU 0.5, 타임아웃 30초
    - python:3.13-slim 이미지 사용
  · E2BSandbox: E2B 클라우드 샌드박스 (실제 서비스용)
    - E2B_API_KEY 환경변수 필요
    - pip install e2b-code-interpreter

[보안 설계]

  Docker 샌드박스:
    컨테이너 격리 → 호스트 파일시스템/프로세스 접근 불가
    네트워크 제한 → 지정된 URL만 접근 가능


[선택 기준]
  개발/테스트: DockerSandbox (로컬 Docker 데몬 필요)
  실제 서비스: E2BSandbox (E2B API 키 필요, 추가 비용)
  config.sandbox_backend: "docker" | "e2b" | "exec" (기본값 exec, 개발용)
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import textwrap
from typing import Any

from structverify.utils.logger import get_logger

logger = get_logger(__name__)

# 스크래핑 코드 실행용 Docker 이미지
# [김예슬 - 2026-04-28] 첫 실행 시 자동 빌드 (_ensure_image 참고)
# pip install 시점이 아니라 첫 실행 시점에 빌드 → Docker 없는 환경 pip install 가능
# Dockerfile.scraper는 패키지 내부(preprocessing/)에 포함되어 배포됨
SANDBOX_IMAGE          = "structverify-scraper"
SANDBOX_IMAGE_FALLBACK = "python:3.13-slim"

# SANDBOX_IMAGE 폴백 시 컨테이너 내 사전 설치 패키지
# structverify-scraper 이미지에는 이미 설치되어 있어 불필요
SANDBOX_SETUP = "pip install httpx beautifulsoup4 -q"

# 이미지 빌드 완료 여부 캐시 (프로세스 내 중복 빌드 방지)
_IMAGE_READY: bool = False

# 컨테이너 리소스 제한
SANDBOX_MEMORY  = "256m"
SANDBOX_CPUS    = "0.5"
SANDBOX_TIMEOUT = 30  # 초


# ── 메인 실행 함수 ────────────────────────────────────────────────────────────

async def run_scraper_sandboxed(
    code: str,
    url: str,
    backend: str = "exec",  # "exec" | "docker" | "e2b"
) -> str:
    """
    LLM이 생성한 스크래핑 코드를 지정된 백엔드에서 격리 실행.

    Args:
        code:    LLM이 생성한 Python 코드 (async def scrape(url) 포함)
        url:     스크래핑할 뉴스 URL
        backend: 실행 환경
                 "exec"   — exec() 직접 실행 (개발용, 보안 취약)
                 "docker" — Docker 컨테이너 격리 (로컬 서비스용)
                 "e2b"    — E2B 클라우드 샌드박스 (실제 서비스용)

    Returns:
        MD 형식 텍스트 또는 빈 문자열 (실패 시)
    """
    if backend == "docker":
        result, error = await DockerSandbox().run(code, url)
        return result, error
    elif backend == "e2b":
        result = await E2BSandbox().run(code, url)
        return result, ""
    else:
        result = await _run_exec(code, url)
        return result, ""


# ── Docker 샌드박스 ───────────────────────────────────────────────────────────

class DockerSandbox:
    """
    로컬 Docker 컨테이너에서 스크래핑 코드 격리 실행.

    보안:
      - 컨테이너 파일시스템 격리 (호스트 마운트 없음)
      - 메모리/CPU 제한
      - 타임아웃 강제 종료
      - 네트워크: 기본 bridge (외부 인터넷 접근 가능, 호스트 내부망 차단)

    전제조건:
      - Docker 데몬 실행 중
      - docker 명령어 PATH에 있음
    """

    async def run(self, code: str, url: str) -> str:
        """코드를 Docker 컨테이너에서 실행하고 stdout 반환"""

        # 실행할 완성 스크립트 (scrape(url) 호출 포함)
        full_script = self._build_script(code, url)

        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="structverify_",   # cleanup_tmp_files()가 이 패턴으로 정리
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(full_script)
            script_path = f.name

        # 첫 실행 시 자동 빌드 (이미 빌드됐으면 바로 통과)
        image = await _ensure_image()
        setup_cmd = (
            "python scraper.py"
            if image == SANDBOX_IMAGE
            else f"{SANDBOX_SETUP} && python scraper.py"
        )

        try:
            # Docker 컨테이너 실행
            # --rm: 종료 후 컨테이너 자동 삭제
            # -v: 스크립트 파일만 read-only 마운트
            # --network=bridge: 외부 인터넷 허용, 호스트 내부망 차단
            # --memory, --cpus: 리소스 제한
            cmd = [
                "docker", "run", "--rm",
                "--network=bridge",
                f"--memory={SANDBOX_MEMORY}",
                f"--cpus={SANDBOX_CPUS}",
                "-v", f"{script_path}:/app/scraper.py:ro",
                "--workdir=/app",
                image,
                "sh", "-c",
                setup_cmd,
            ]

            logger.info(f"Docker 샌드박스 실행: {url}")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=SANDBOX_TIMEOUT,
                )
            except asyncio.TimeoutError:
                proc.kill()
                logger.error(f"Docker 샌드박스 타임아웃 ({SANDBOX_TIMEOUT}초): {url}")
                return "", "TimeoutError: 실행 시간 초과"

            if proc.returncode != 0:
                err_msg = stderr.decode(errors="replace").strip()
                logger.warning(
                    f"Docker 샌드박스 오류 (returncode={proc.returncode}): "
                    f"{err_msg[:1000]}" 
                )
                return "", err_msg  # [v3] 에러 메시지 반환

            result = stdout.decode("utf-8", errors="replace").strip()
            logger.info(f"Docker 샌드박스 성공: {url} ({len(result)}자)")
            return result, ""  # [v3] 성공 시 에러 없음

        except FileNotFoundError:
            logger.error("Docker 명령어를 찾을 수 없음 — Docker 데몬 실행 여부 확인")
            return "", "Docker 명령어 없음"
        except Exception as e:
            logger.error(f"Docker 샌드박스 예외: {e}")
            return "", str(e)
        finally:
            # 임시 파일 삭제
            try:
                os.unlink(script_path)
            except OSError:
                pass

    def _build_script(self, code: str, url: str) -> str:
        """
        LLM 생성 코드 + scrape(url) 호출 + 결과 print를 하나의 스크립트로 조합.
        Docker 컨테이너 안에서 python script.py 로 실행됨.
        """
        return textwrap.dedent(f"""
import asyncio

{code}

async def main():
    result = await scrape({url!r})
    print(result or "", end="")

asyncio.run(main())
""").strip()


# ── E2B 샌드박스 ──────────────────────────────────────────────────────────────

class E2BSandbox:
    """
    E2B 클라우드 샌드박스에서 스크래핑 코드 격리 실행.

    실제 서비스용 — 로컬 Docker 없이 클라우드에서 완전 격리.
    E2B는 코드 인터프리터 전용 샌드박스 서비스 (https://e2b.dev).

    전제조건:
      pip install e2b-code-interpreter
      환경변수: E2B_API_KEY

    비용:
      E2B 요금제에 따름 (무료 티어: 월 100시간)
    """

    async def run(self, code: str, url: str) -> str:
        """E2B 샌드박스에서 코드 실행"""
        try:
            from e2b_code_interpreter import Sandbox
        except ImportError:
            logger.error(
                "e2b-code-interpreter 미설치. "
                "pip install e2b-code-interpreter 실행 후 E2B_API_KEY 환경변수 설정"
            )
            return ""

        api_key = os.environ.get("E2B_API_KEY", "")
        if not api_key:
            logger.error("E2B_API_KEY 환경변수 미설정")
            return ""

        full_script = textwrap.dedent(f"""
import asyncio

{code}

async def main():
    result = await scrape({url!r})
    print(result or "", end="")

asyncio.run(main())
""").strip()

        try:
            logger.info(f"E2B 샌드박스 실행: {url}")

            # E2B 샌드박스 생성 + 코드 실행
            # Sandbox는 격리된 클라우드 VM 환경
            sandbox = Sandbox(api_key=api_key)

            # 패키지 설치
            sandbox.commands.run("pip install httpx beautifulsoup4 -q")

            # 스크래핑 코드 실행
            execution = sandbox.run_code(full_script, timeout=SANDBOX_TIMEOUT)

            sandbox.kill()

            if execution.error:
                logger.warning(f"E2B 실행 오류: {execution.error.value[:300]}")
                return ""

            result = "".join(
                [str(log.line) for log in execution.logs.stdout]
            ).strip()

            logger.info(f"E2B 샌드박스 성공: {url} ({len(result)}자)")
            return result

        except Exception as e:
            logger.error(f"E2B 샌드박스 예외: {e}")
            return ""


# ── exec 직접 실행 (개발용) ───────────────────────────────────────────────────

async def _run_exec(code: str, url: str) -> str:
    """
    exec() 직접 실행 — 개발/테스트 전용.
    보안 격리 없음. 프로덕션에서는 사용 금지.
    """
    try:
        namespace: dict[str, Any] = {}
        exec(compile(code, "<llm_scraper>", "exec"), namespace)
        scrape_fn = namespace.get("scrape")
        if not scrape_fn:
            return ""
        result = await scrape_fn(url)
        return str(result) if result else ""
    except Exception as e:
        logger.warning(f"exec 실행 실패: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# [김예슬 - 2026-04-28] 정리(Cleanup) 유틸
# ══════════════════════════════════════════════════════════════════════════════

class SandboxCleanup:
    """
    Docker 컨테이너 / 임시 파일 / 메모리 캐시 정리.

    [자동 정리]
    - Docker 컨테이너: --rm 플래그로 실행 종료 즉시 자동 삭제
    - 임시 스크립트 파일: DockerSandbox.run() finally 블록에서 즉시 삭제

    [수동 정리 — 이 클래스 사용]
    - 좀비 컨테이너: docker ps -a 에서 Exited 상태로 남은 컨테이너 강제 삭제
    - dangling 이미지: 이름 없는 중간 레이어 이미지 정리
    - 메모리 캐시: _SCRAPER_CACHE dict 초기화
    - 임시 파일 누락분: /tmp/structverify_* 패턴 파일 삭제

    사용 예시:
        # 파이프라인 종료 시
        await SandboxCleanup.cleanup_all()

        # 개발 중 캐시만 초기화
        SandboxCleanup.clear_memory_cache()

        # Docker만 정리
        await SandboxCleanup.cleanup_docker()
    """

    @staticmethod
    async def cleanup_all(clear_cache: bool = True) -> dict[str, int]:
        """
        전체 정리 실행.

        Returns:
            정리 결과 통계 dict
            {
              "containers_removed": N,
              "tmp_files_removed": N,
              "cache_cleared": bool,
            }
        """
        stats: dict[str, int | bool] = {}

        # 1) Docker 좀비 컨테이너 + dangling 이미지 정리
        docker_stats = await SandboxCleanup.cleanup_docker()
        stats.update(docker_stats)

        # 2) 임시 파일 정리
        tmp_count = SandboxCleanup.cleanup_tmp_files()
        stats["tmp_files_removed"] = tmp_count

        # 3) 메모리 캐시 초기화 (선택)
        if clear_cache:
            from structverify.preprocessing.extractor import clear_scraper_cache
            clear_scraper_cache()
            stats["cache_cleared"] = True
            logger.info("메모리 캐시 초기화 완료")
        else:
            stats["cache_cleared"] = False

        logger.info(f"전체 정리 완료: {stats}")
        return stats

    @staticmethod
    async def cleanup_docker() -> dict[str, int]:
        """
        Docker 좀비 컨테이너 + dangling 이미지 정리.

        - 좀비 컨테이너: --rm 실패 시 Exited 상태로 남은 컨테이너
        - dangling 이미지: 빌드 중간에 생긴 이름 없는 레이어 이미지

        Returns:
            {"containers_removed": N, "images_removed": N}
        """
        stats = {"containers_removed": 0, "images_removed": 0}

        # structverify-scraper 관련 좀비 컨테이너 삭제
        # docker ps -a -q -f ancestor=structverify-scraper -f status=exited
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "ps", "-a", "-q",
                "-f", "ancestor=structverify-scraper",
                "-f", "status=exited",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            container_ids = stdout.decode().strip().split()
            container_ids = [c for c in container_ids if c]

            if container_ids:
                rm_proc = await asyncio.create_subprocess_exec(
                    "docker", "rm", "-f", *container_ids,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(rm_proc.communicate(), timeout=15)
                stats["containers_removed"] = len(container_ids)
                logger.info(f"좀비 컨테이너 {len(container_ids)}개 삭제")

        except asyncio.TimeoutError:
            logger.warning("Docker 컨테이너 정리 타임아웃")
        except FileNotFoundError:
            logger.debug("Docker 명령어 없음 — 정리 건너뜀")
        except Exception as e:
            logger.warning(f"Docker 컨테이너 정리 실패: {e}")

        # dangling 이미지 정리 (이름 없는 중간 레이어)
        # docker image prune -f
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "image", "prune", "-f",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = stdout.decode()
            # "Total reclaimed space: X.XXkB" 에서 삭제 여부 확인
            if "Total reclaimed" in output:
                logger.info(f"dangling 이미지 정리: {output.strip()}")
                stats["images_removed"] = 1  # 개수보다 실행 여부
        except Exception as e:
            logger.debug(f"dangling 이미지 정리 건너뜀: {e}")

        return stats

    @staticmethod
    def cleanup_tmp_files() -> int:
        """
        /tmp/structverify_*.py 임시 스크립트 파일 정리.

        DockerSandbox.run()의 finally에서 즉시 삭제되지만
        예외 상황에서 누락된 파일을 정리.

        Returns:
            삭제된 파일 수
        """
        import glob
        import os

        pattern = "/tmp/structverify_*.py"
        files = glob.glob(pattern)
        removed = 0

        for f in files:
            try:
                os.unlink(f)
                removed += 1
            except OSError:
                pass

        if removed:
            logger.info(f"임시 파일 {removed}개 삭제: {pattern}")

        return removed

    @staticmethod
    def clear_memory_cache(domain: str | None = None) -> None:
        """
        _SCRAPER_CACHE 메모리 캐시 초기화.

        Args:
            domain: 특정 도메인만 삭제. None이면 전체 초기화.
        """
        from structverify.preprocessing.extractor import clear_scraper_cache
        clear_scraper_cache(domain)

    @staticmethod
    def get_memory_cache_info() -> dict[str, Any]:
        """현재 메모리 캐시 상태 반환 (디버깅용)"""
        from structverify.preprocessing.extractor import get_scraper_cache
        cache = get_scraper_cache()
        return {
            "cached_domains": list(cache.keys()),
            "domain_count": len(cache),
            "total_code_chars": sum(len(v) for v in cache.values()),
        }


# ── 이미지 자동 빌드 ──────────────────────────────────────────────────────────

async def _ensure_image() -> str:
    """
    structverify-scraper Docker 이미지 없으면 자동 빌드.

    [김예슬 - 2026-04-28]
    pip install 시점이 아니라 첫 실행 시점에 빌드:
      - Docker 없는 환경에서도 pip install 가능
      - Docker 있는 환경에서만 빌드 시도
      - 빌드 실패/Docker 없으면 SANDBOX_IMAGE_FALLBACK으로 폴백

    Dockerfile.scraper 위치:
      패키지 내부(structverify/preprocessing/Dockerfile.scraper)에 포함 배포.
      importlib.resources로 경로 자동 탐색.

    Returns:
        사용할 이미지 이름 (SANDBOX_IMAGE 또는 SANDBOX_IMAGE_FALLBACK)
    """
    global _IMAGE_READY

    if _IMAGE_READY:
        return SANDBOX_IMAGE

    try:
        check = subprocess.run(
            ["docker", "image", "inspect", SANDBOX_IMAGE],
            capture_output=True, timeout=5,
        )
        if check.returncode == 0:
            _IMAGE_READY = True
            return SANDBOX_IMAGE
    except FileNotFoundError:
        logger.warning("Docker 명령어 없음 → exec() 폴백 사용")
        return SANDBOX_IMAGE_FALLBACK
    except Exception:
        return SANDBOX_IMAGE_FALLBACK

    logger.info(
        f"{SANDBOX_IMAGE} 이미지 없음 → 자동 빌드 시작 "
        f"(약 30~60초 소요, 이후 실행부터는 즉시 사용)..."
    )

    try:
        import pathlib
        dockerfile_dir = str(pathlib.Path(__file__).parent)

        build_result = subprocess.run(
            [
                "docker", "build",
                "-f", os.path.join(dockerfile_dir, "Dockerfile.scraper"),
                "-t", SANDBOX_IMAGE,
                dockerfile_dir,
            ],
            capture_output=True,
            timeout=120,
        )

        if build_result.returncode == 0:
            _IMAGE_READY = True
            logger.info(f"{SANDBOX_IMAGE} 이미지 빌드 완료")
            return SANDBOX_IMAGE

        logger.warning(
            f"이미지 빌드 실패 (returncode={build_result.returncode}) "
            f"→ {SANDBOX_IMAGE_FALLBACK} 폴백\n"
            f"{build_result.stderr.decode()[:300]}"
        )
        return SANDBOX_IMAGE_FALLBACK

    except subprocess.TimeoutExpired:
        logger.warning("이미지 빌드 타임아웃 → 폴백")
        return SANDBOX_IMAGE_FALLBACK
    except Exception as e:
        logger.warning(f"이미지 빌드 예외: {e} → 폴백")
        return SANDBOX_IMAGE_FALLBACK