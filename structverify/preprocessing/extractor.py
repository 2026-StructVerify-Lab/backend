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
"""
from __future__ import annotations

from structverify.core.schemas import SourceType
from structverify.preprocessing.pdf import extract_pdf_to_markdown
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


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
    raise ValueError(f"미지원 소스: {source_type}")


async def _extract_from_url(url: str) -> str:
    """
    URL → 본문 텍스트 추출 (Trafilatura)
    TODO[DONE]: trafilatura.fetch_url() + trafilatura.extract(include_tables=True) 구현
    """
    downloaded = trafilatura.fetch_url(url)
    result = trafilatura.extract(downloaded, # 본문 및 메타데이터 추출 / 1차: json
                                 output_format="json", 
                                 include_comments=False, 
                                 include_links=False, 
                                 with_metadata=True,
                                 include_tables=True)
    parsed_result = json.loads(result) # JSON 문자열을 파이썬 딕셔너리로 변환
    markdown = f"# {parsed_result.get('title', '')}\n\n{parsed_result.get('text', '')}" # 제목과 본문을 마크다운 형식으로 결합
    logger.info(f"URL extracted successfully: {url}")
    return markdown
    # logger.warning(f"URL 추출 stub: {url}")
    # return f"[STUB] {url}"


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
