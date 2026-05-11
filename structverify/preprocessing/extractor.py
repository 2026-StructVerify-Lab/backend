"""
preprocessing/extractor.py — 소스 유형별 텍스트 추출기 (디스패처)

세부 구현은 유형별 하위 모듈에 위임:
  * PDF  → `preprocessing.pdf`  (PyMuPDF + Docling + OCR 파이프라인, Markdown 반환)
  * URL  → Trafilatura (TODO)
  * DOCX → python-docx (TODO) << 우선순위 후순위

[참고] Trafilatura (Barbaresi, ACL 2021) — https://github.com/adbar/trafilatura
  논문: Barbaresi, A. (2021). Trafilatura: A Web Scraping Library and
  Command-Line Tool for Text Discovery and Extraction. ACL 2021.
  URL 입력 시 광고/네비게이션 제거 후 본문만 추출

[참고] Docling (IBM Research, 2024) — https://github.com/DS4SD/docling
  PDF/DOCX를 통합 DoclingDocument JSON으로 변환. TableFormer로 테이블 구조 인식.

[참고] PyMuPDF — https://github.com/pymupdf/PyMuPDF
  PDF 텍스트/테이블 페이지 단위 추출

[참고] python-docx — https://github.com/python-openxml/python-docx
  DOCX XML 구조 파싱
  
[0422(수) 진행 내용 관련 참고 논문]
[참고] TableFormer (Nassar et al., IBM Research, CVPR 2022) — https://arxiv.org/abs/2203.01017
  논문: Nassar, A., Livathinos, N., Lysak, M., & Staar, P. (2022).
  TableFormer: Table Structure Understanding with Transformers. CVPR 2022.
  Docling 내부의 테이블 인식 모델. HTML 표 구조 보존의 근거.

[참고] Docling Technical Report (Livathinos et al., IBM Research, 2024) — https://arxiv.org/abs/2408.09869
  논문: Auer, C., Lysak, M., Nassar, A., Dolfi, M., Livathinos, N., et al. (2024).
  Docling Technical Report. arXiv:2408.09869.
  PDF→DoclingDocument 변환 파이프라인 전체 구조. JSON/HTML export 의 기반.

[참고] CRAFT (Baek et al., NAVER, CVPR 2019) — https://arxiv.org/abs/1904.01941
  논문: Baek, Y., Lee, B., Han, D., Yun, S., & Lee, H. (2019).
  Character Region Awareness for Text Detection. CVPR 2019.
  EasyOCR 의 텍스트 영역 검출(detector) 모듈 백본.

[참고] CRNN (Shi et al., TPAMI 2017) — https://arxiv.org/abs/1507.05717
  논문: Shi, B., Bai, X., & Yao, C. (2017).
  An End-to-End Trainable Neural Network for Image-based Sequence Recognition
  and its Application to Scene Text Recognition. IEEE TPAMI.
  EasyOCR 의 텍스트 인식(recognizer) 모듈 백본 (CNN+RNN+CTC).

[참고] Tesseract OCR (Smith, Google, ICDAR 2007) — https://github.com/tesseract-ocr/tesseract
  논문: Smith, R. (2007). An Overview of the Tesseract OCR Engine. ICDAR 2007.
  한국어 성능은 EasyOCR 에 밀리지만 영문·속도 기준 벤치 기준점으로 포함.

[참고] PP-OCR (Du et al., Baidu, 2020) — https://arxiv.org/abs/2009.09941
  논문: Du, Y., Li, C., Guo, R., Yin, X., Liu, W., et al. (2020).
  PP-OCR: A Practical Ultra Lightweight OCR System. arXiv:2009.09941.
  PaddleOCR 백엔드의 기반 모델. `OCR_BACKEND=paddle` 로 활성화 가능.

[김예슬 - 2026-04-28 / v3]
- LLMScraper 클래스 추가 (trafilatura 실패 시 LLM 동적 스크래핑)
  · trafilatura 1차 시도 → 실패/200자 미만이면 LLMScraper 2차 시도
  · LLM(HCX-003)이 사이트 HTML 샘플 분석 후 스크래핑 코드 동적 생성
  · 생성된 코드를 Docker 컨테이너에서 격리 실행 (sandbox_backend="docker" 기본값)
  · 도메인별 성공 코드를 _SCRAPER_CACHE에 저장 → 같은 사이트 재요청 시 재사용
  · 실패 시 에러 메시지를 LLM에 피드백 → 코드 수정 (Self-Refine, 최대 2회)
- _extract_from_url() 변경:
  · v2: trafilatura만 사용 (실패 시 예외 발생)
  · v3: _try_trafilatura() 1차 → 실패/짧으면 LLMScraper 2차
        어떤 경우든 Step 2~9 진행 가능한 텍스트 반환
- _try_trafilatura(): 기존 v2 로직을 별도 함수로 분리 (실패 시 빈 문자열)
- get_scraper_cache(), clear_scraper_cache(): 캐시 관리 유틸 추가

[설계 원칙 - v3]
- sandbox_backend 기본값 "docker" (보안 격리)
  · Docker 없는 환경: _ensure_image() fallback → exec() 직접 실행
- LLMScraper는 Agent 행동이나 전처리(Step 1)에 귀속
  · 추후 builder_agent.pretrain_domain()에서 주요 사이트 사전 크롤링 이관 예정
"""
from __future__ import annotations

# [v3 김예슬] 추가 import
import re
from typing import Any
from urllib.parse import urlparse
import httpx
import trafilatura, json
from bs4 import BeautifulSoup

from structverify.core.schemas import SourceType
from structverify.preprocessing.pdf import extract_pdf_to_markdown
from structverify.utils.logger import get_logger

logger = get_logger(__name__)



_DOMAIN_SCRAPERS = {
    "chosun.com": "_scrape_chosun_fixed",
}

# ── [v3 김예슬] 도메인별 스크래핑 코드 캐시 ──────────────────────────────────
# key: domain (예: "chosun.com"), value: 실행 가능한 Python 코드 문자열
# 프로세스 재시작 시 초기화 — 추후 scraper_cache.yaml로 영속화 예정

_SCRAPER_CACHE: dict[str, str] = {}

# [v3 김예슬] LLM 스크래퍼 코드 생성 프롬프트
_SCRAPER_GEN_PROMPT = """당신은 Python 웹 스크래핑 전문가입니다.
아래 뉴스 기사 URL에서 본문 텍스트를 추출하는 Python 코드를 작성하세요.

URL: {url}
사이트: {domain}
HTML 샘플 (앞 3000자):
{html_sample}

[요구사항]
- httpx와 BeautifulSoup4만 사용 (import httpx, from bs4 import BeautifulSoup)
- async def scrape(url: str) -> str: 함수로 작성
- 반환값: "# 제목\n\n본문 텍스트" 형식의 마크다운 문자열
- 본문은 .get_text(separator="\n", strip=True)로 HTML 태그 완전 제거
- 광고, 네비게이션, 댓글, 저작권 문구 제거
- User-Agent 헤더 포함 (Mozilla/5.0 Chrome 계열)
- 실패 시 빈 문자열 반환 (예외 발생 금지)
- HTML 태그(<p>, <div>, <span> 등)가 결과에 포함되면 안 됨

[주의]
- 코드만 반환 (설명 없이)
- ```python 코드블록 없이 순수 코드만
- import 문 포함

Python 코드:"""

# [v3 김예슬] 에러 피드백 기반 코드 수정 프롬프트 (Self-Refine)
_SCRAPER_REFINE_PROMPT = """앞서 작성한 스크래핑 코드가 아래 에러로 실패했습니다.
에러를 분석하고 코드를 수정하세요.

URL: {url}
사이트: {domain}

HTML 샘플 (앞 2000자):
{html_sample}

실패한 코드:
{failed_code}

에러:
{error}

[수정 방향]
- 에러 원인을 분석해서 해당 부분만 수정
- 조선일보 등 Next.js 사이트는 script 태그 JSON에서 본문 추출 시도
  예: soup.find_all("script")[0].string → json.loads() → description/articleBody 필드
- try/except로 각 selector 실패를 안전하게 처리
- 반환값에 HTML 태그가 포함되면 안 됨 (.get_text()로 텍스트만 추출)
- 결과 형식: "# 제목\n\n순수 텍스트 본문" (마크다운)

수정된 코드만 반환하세요 (설명 없이):"""


async def extract_text(source: str, source_type: SourceType) -> str:
    """소스 유형에 따라 적절한 추출기를 호출하여 텍스트 반환"""
    if source_type == SourceType.URL:
        return await _extract_from_url(source)
    elif source_type == SourceType.PDF:
        return _extract_from_pdf(source)
    elif source_type == SourceType.DOCX:
        return _extract_from_docx(source)
    elif source_type == SourceType.TEXT:
        return source


async def _extract_from_url(url: str) -> str:
    """
    URL → 본문 텍스트 추출

    [v2] trafilatura만 사용 (실패 시 예외)
    [v3 김예슬] trafilatura 1차 → 실패/200자 미만이면 LLMScraper 2차
      - 조선일보, 구독제 사이트 등 trafilatura 차단 사이트 대응
      - LLMScraper가 Docker 격리 환경에서 코드 실행 (sandbox_backend="docker" 기본)
      - 두 단계 모두 실패해도 Step 2~9 진행 가능한 텍스트 반환
    """
    # [v2] trafilatura 기존 로직 — _try_trafilatura()로 분리
    result = _try_trafilatura(url)
    if result and len(result.strip()) > 200:
        logger.info(f"URL extracted successfully: {url}")
        return result

    # [v3 김예슬] trafilatura 실패/짧음 → LLMScraper 2차 시도
    logger.warning(f"trafilatura 실패/짧음 → LLMScraper 시도: {url}")
    scraper = LLMScraper()
    result = await scraper.scrape(url)

    if result and len(result.strip()) > 200:
        logger.info(f"LLMScraper 성공: {url} ({len(result)}자)")
        return result

    logger.error(f"URL 추출 완전 실패: {url}")
    return f"# 추출 실패\n\nURL: {url} — 본문을 가져올 수 없습니다."


def _try_trafilatura(url: str) -> str:
    """
    [v3 김예슬] v2 trafilatura 로직을 별도 함수로 분리.
    실패 시 빈 문자열 반환 (예외 발생 안 함).

    v2 원본 코드 (주석 보존):
        downloaded = trafilatura.fetch_url(url)
        result = trafilatura.extract(downloaded, output_format="json", ...)
        parsed_result = json.loads(result)
        markdown = f"# {parsed_result.get('title', '')}\\n\\n{parsed_result.get('text', '')}"
        logger.info(f"URL extracted successfully: {url}")
        return markdown
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        result = trafilatura.extract(
            downloaded,              # 본문 및 메타데이터 추출 / 1차: json
            output_format="json",
            include_comments=False,
            include_links=False,
            with_metadata=True,
            include_tables=True,
        )
        if not result:
            return ""
        parsed_result = json.loads(result)   # JSON 문자열을 파이썬 딕셔너리로 변환
        markdown = f"# {parsed_result.get('title', '')}\n\n{parsed_result.get('text', '')}"  # 제목과 본문을 마크다운 형식으로 결합
        return markdown
    except Exception as e:
        logger.debug(f"trafilatura 예외: {e}")
        return ""


def _extract_from_pdf(filepath: str) -> str:
    """
    PDF → Markdown 추출.
    구현은 `structverify.preprocessing.pdf` 하위 패키지에 위임.
    파이프라인: PyMuPDF 병렬 텍스트 → Docling JSON/HTML 스코어링 →
               title/date/body 추출 → 이미지 crop OCR 인라인 → 스캔 OCR → Markdown
    TODO[DONE]: fitz.open(filepath) → page.get_text("text") 구현
    TODO[DONE]: 스캔 PDF OCR 폴백 (Tesseract / EasyOCR)
    TODO[DONE]: Docling 연동으로 테이블 구조 인식 강화
    """
    return extract_pdf_to_markdown(filepath)


def _extract_from_docx(filepath: str) -> str:
    """
    DOCX → 문단/테이블 추출 (python-docx)
    TODO: Document(filepath).paragraphs + .tables 파싱 구현
    """
    logger.warning(f"DOCX 추출 stub: {filepath}")
    return f"[STUB] {filepath}"


# ══════════════════════════════════════════════════════════════════════════════
# [v3 김예슬] LLMScraper — trafilatura 실패 시 LLM 동적 스크래핑
# ══════════════════════════════════════════════════════════════════════════════

class LLMScraper:
    """
    LLM이 사이트별 스크래핑 코드를 동적으로 생성하고 Docker에서 격리 실행.

    [v3 김예슬 - 2026-04-28]

    흐름:
      1) URL에서 도메인 추출 (예: chosun.com)
      2) _SCRAPER_CACHE에 해당 도메인 코드 있으면 Docker에서 바로 실행
      3) 없으면:
         a) httpx로 raw HTML 가져오기 (앞 3000자만)
         b) LLM(HCX-003)에게 HTML 샘플 전달 → 스크래핑 코드 생성
         c) Docker 컨테이너 격리 실행 (sandbox_backend="docker" 기본)
         d) 성공하면 _SCRAPER_CACHE[domain] = code 저장
         e) 실패하면 에러 메시지를 LLM에 피드백 → 코드 수정 (Self-Refine, 최대 2회)
    """

    def __init__(self, config: dict | None = None):
        self.config    = config or {}
        self.max_retry = 2
        self.timeout   = 30
        
        
   
    async def _refine_scraper_code(
        self, url: str, domain: str, failed_code: str, error: str,
        html: str = "",
    ) -> str:
        """
        [v3 김예슬] Self-Refine — 에러 피드백으로 스크래핑 코드 수정.

        실패한 코드 + 에러 메시지 + HTML 샘플을 LLM에 전달 → 수정된 코드 반환.
        조선일보 등 Next.js 사이트의 경우 script JSON 파싱 방향으로 유도.
        """
        from structverify.utils.llm_client import LLMClient
        llm = LLMClient(config=self.config.get("llm", {}))

        prompt = _SCRAPER_REFINE_PROMPT.format(
            url=url,
            domain=domain,
            html_sample=html[:2000],      # [v2에서 누락] html_sample 추가
            failed_code=failed_code[:2000],
            error=error[:500],
        )
        try:
            code = await llm.generate(
                prompt=prompt,
                system_prompt="Python 웹 스크래핑 전문가. 에러를 분석해서 코드를 수정하세요.",
                model_tier="heavy",
            )
            code = re.sub(r"```python\s*", "", code)
            code = re.sub(r"```\s*", "", code)
            return code.strip()
        except Exception as e:
            logger.error(f"코드 수정 실패: {e}")
            return ""

    async def _run_code(self, code: str, url: str) -> tuple[str, str]:
        """
        [v3 김예슬] 생성된 코드를 sandbox_backend에 따라 격리 실행.

        sandbox_backend 기본값 "docker" (보안 격리).
          "docker" — Docker 컨테이너 격리 실행 (기본)
          "e2b"    — E2B 클라우드 샌드박스
          "exec"   — exec() 직접 실행 (개발용, 보안 취약)

        Returns:
            (result, error) — 성공 시 error="", 실패 시 result=""

        Docker 없는 환경: _ensure_image() fallback → exec() 직접 실행
        """
        from structverify.preprocessing.scraper_sandbox import run_scraper_sandboxed
        backend = self.config.get("sandbox_backend", "docker")  # 기본값 docker
        return await run_scraper_sandboxed(code, url, backend=backend)


    async def _generate_scraper_code(self, url: str, domain: str, html: str) -> str:
        """LLM(HCX-003)에게 HTML 샘플을 주고 스크래핑 코드 생성 요청"""
        from structverify.utils.llm_client import LLMClient
        llm = LLMClient(config=self.config.get("llm", {}))

        # [v2에서 버그] _SCRAPER_REFINE_PROMPT 잘못 사용 → _SCRAPER_GEN_PROMPT로 수정
        prompt = _SCRAPER_GEN_PROMPT.format(
            url=url,
            domain=domain,
            html_sample=html[:3000],
        )
        try:
            code = await llm.generate(
                prompt=prompt,
                system_prompt="Python 웹 스크래핑 전문가. 실행 가능한 코드만 반환.",
                model_tier="heavy",  # HCX-003 — 코드 생성 정확도 중요
            )
            # 코드블록 마크다운 제거
            code = re.sub(r"```python\s*", "", code)
            code = re.sub(r"```\s*", "", code)
            return code.strip()
        except Exception as e:
            logger.error(f"LLM 코드 생성 실패: {e}")
            return ""
          
    async def scrape(self, url: str) -> str:
        """URL → MD 형식 텍스트 (캐시 히트 또는 LLM 생성 코드 실행)"""
        domain = _extract_domain(url)
        
        fixed_result = await _run_domain_scraper(domain, url)
        if fixed_result and len(fixed_result.strip()) > 200:
            logger.info(f"도메인 고정 스크래퍼 성공: {domain}")
            return fixed_result

        # 캐시 히트 → Docker에서 바로 실행
        if domain in _SCRAPER_CACHE:
            logger.info(f"스크래퍼 캐시 히트: {domain}")
            result, _ = await self._run_code(_SCRAPER_CACHE[domain], url)
            if result and len(result.strip()) > 200:
                return result
            logger.warning(f"캐시 코드 실패 → 재생성: {domain}")
            del _SCRAPER_CACHE[domain]

        # raw HTML 가져오기
        html = await _fetch_raw_html(url, self.timeout)
        if not html:
            logger.error(f"HTML 가져오기 실패: {url}")
            return ""

        # 초기 코드 생성
        code = await self._generate_scraper_code(url, domain, html)
        if not code:
            return ""

        last_error = ""
        for attempt in range(1, self.max_retry + 1):
            if attempt > 1 and last_error:
                # [v3 Self-Refine] 에러 메시지를 LLM에 피드백해서 코드 수정
                logger.info(
                    f"에러 피드백 → 코드 수정 (시도 {attempt}): {domain} | "
                    f"에러: {last_error[:200]}"
                )
                code = await self._refine_scraper_code(url, domain, code, last_error, html=html)
                if not code:
                    break

            result, error = await self._run_code(code, url)

            if result and len(result.strip()) > 200:
                _SCRAPER_CACHE[domain] = code
                logger.info(f"스크래퍼 캐시 저장: {domain}")
                return result

            # 빈 결과도 에러로 처리 → LLM 피드백에 활용
            if not error and (not result or len(result.strip()) <= 200):
                error = (
                    f"스크래핑 결과가 비어있습니다 ({len(result or '')}자). "
                    f"p태그 0개, article태그 없음. "
                    f"soup.find_all('script')[0].string을 json.loads()로 파싱 후 "
                    f"description 또는 articleBody 필드를 사용하세요."
                )

            last_error = error
            logger.warning(f"스크래퍼 코드 실행 실패 (시도 {attempt}): {domain}")

        return ""


# ── 도메인별 고정 스크래퍼 실행 ─────────────────────────

async def _run_domain_scraper(domain: str, url: str) -> str:
    try:
        if "chosun.com" in domain:
            result = await _scrape_chosun_fixed(url)

            if result and len(result.strip()) > 200:
                return result

        return ""

    except Exception as e:
        logger.warning(f"도메인 스크래퍼 실패 → LLM scraper fallback: {domain} — {e}")
        return ""

async def _scrape_chosun_fixed(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.9",
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    }

    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            res = await client.get(url, headers=headers)
            res.raise_for_status()
            html = res.text
    except Exception as e:
        logger.warning(f"조선일보 HTML 요청 실패: {e}")
        return ""

    soup = BeautifulSoup(html, "html.parser")

    title = ""
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()
    elif soup.title:
        title = soup.title.get_text(strip=True)

    # 1) 브라우저 렌더링 후 DOM에 본문이 있는 경우
    container = soup.select_one("section.article-body[itemprop='articleBody'], section.article-body")
    if container:
        paragraphs = container.select("p.article-body__content")
        texts = [
            p.get_text(separator=" ", strip=True)
            for p in paragraphs
            if p.get_text(strip=True)
        ]
        body = _clean_article_text("\n\n".join(texts))

        logger.info(f"[chosun] article-body paragraphs={len(paragraphs)}, text_len={len(body)}")

        if len(body) > 200:
            return f"# {title}\n\n{body}"

    # 2) httpx 원본 HTML에 들어있는 Fusion.globalContent.content_elements 파싱
    fusion_body = _extract_chosun_fusion_body(html)
    if fusion_body and len(fusion_body.strip()) > 200:
        logger.info(f"[chosun] Fusion.globalContent body hit, text_len={len(fusion_body)}")
        return f"# {title}\n\n{fusion_body}"

    logger.warning("[chosun] section.article-body / Fusion.globalContent 본문 추출 실패")
    return ""
  
  
def _extract_chosun_fusion_body(html: str) -> str:
    """
    조선일보 원본 HTML의 Fusion.globalContent.content_elements에서
    type='text' 항목의 content를 추출한다.
    """
    match = re.search(
        r"Fusion\.globalContent\s*=\s*(\{.*?\});Fusion\.globalContentConfig",
        html,
        flags=re.DOTALL,
    )
    if not match:
        logger.warning("[chosun] Fusion.globalContent JSON not found")
        return ""

    raw_json = match.group(1)

    try:
        data = json.loads(raw_json)
    except Exception as e:
        logger.warning(f"[chosun] Fusion.globalContent JSON parse failed: {e}")
        return ""

    elements = data.get("content_elements", [])
    texts = []

    for el in elements:
        if not isinstance(el, dict):
            continue
        if el.get("type") != "text":
            continue

        content = el.get("content", "")
        if not content:
            continue

        text = _html_to_plain_text(content)
        if text:
            texts.append(text)

    body = "\n\n".join(texts)
    return _clean_article_text(body)


def _html_to_plain_text(fragment: str) -> str:
    """HTML fragment를 순수 텍스트로 변환"""
    soup = BeautifulSoup(fragment, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return text.strip()
  
def _clean_article_text(text: str) -> str:
    """기사 본문 텍스트 정리"""
    if not text:
        return ""

    # HTML escape / 공백 정리
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # 조선일보/언론사 공통 잡문구 제거
    remove_patterns = [
        r"Copyright.*?reserved\.",
        r"무단 전재.*?재배포 금지",
        r"기자\s*$",
        r"구독.*?신청",
        r"좋아요.*?공유",
    ]

    for pattern in remove_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    return text.strip()

# ── [v3 김예슬] 내부 헬퍼 ────────────────────────────────────────────────────

def _extract_domain(url: str) -> str:
    """URL에서 도메인 추출. 예: 'www.chosun.com' → 'chosun.com'"""
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


async def _fetch_raw_html(url: str, timeout: int = 30) -> str:
    """URL에서 raw HTML 가져오기 (크롤링 차단 우회 User-Agent 포함)"""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml,*/*;q=0.9",
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    }
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.text
    except Exception as e:
        logger.error(f"HTML 가져오기 실패: {url} — {e}")
        return ""


# ── [v3 김예슬] 캐시 관리 유틸 ───────────────────────────────────────────────

def get_scraper_cache() -> dict[str, str]:
    """현재 캐시 상태 반환 (디버깅/테스트용)"""
    return dict(_SCRAPER_CACHE)


def clear_scraper_cache(domain: str | None = None) -> None:
    """캐시 초기화. domain 지정 시 해당 도메인만 삭제"""
    if domain:
        _SCRAPER_CACHE.pop(domain, None)
        logger.info(f"캐시 삭제: {domain}")
    else:
        _SCRAPER_CACHE.clear()
        logger.info("전체 캐시 초기화")