"""
detection/domain_classifier.py — 도메인 자동 분류 (Step 3)

입력 텍스트의 도메인을 판별하고 적절한 Domain Pack을 선택한다.

[김예슬 - 2026-04-22]
- DOMAIN_CLASSIFY_PROMPT: few-shot 예시 3개 포함한 프롬프트로 교체
- classify_domain(): model_tier="light"(HCX-DASH-001)로 명시적 지정
- _build_text_preview(): 블록 타입 고려한 미리보기 텍스트 구성
- _load_domain_pack(): domain-packs/{domain}/prompts.yaml 로드 시도

[김예슬 - 2026-04-23 v1]
- SUPPORTED_DOMAINS 하드코딩 제거 → domain-packs/ 디렉토리 기반으로 변경

[김예슬 - 2026-04-23 v1]
- DomainRegistry 클래스 추가: 레지스트리 기반 도메인 관리
  · 문제: LLM 자유 생성 시 같은 주제를 다른 이름으로 분류하는 파편화 발생
    (예: real_estate / housing_market / property → 모두 같은 주제)
  · 해결: 기존 등록 도메인 목록 + 설명을 LLM 프롬프트에 주입
    → LLM이 기존 도메인 중 유사한 게 있으면 재사용, 없으면 신규 생성
  · 신규 도메인 생성 시 레지스트리(registry.yaml)에 자동 저장
- classify_domain() 반환값: str → tuple[str, str] (domain, description)
  · domain: 도메인 키 (영문 소문자)
  · description: 도메인 한국어 설명 (레지스트리에서 조회 또는 신규 생성)
- DOMAIN_CLASSIFY_PROMPT: 기존 도메인 목록 동적 주입 방식으로 변경

[참고] ReAct (Yao et al., ICLR 2023)
  Agent의 첫 단계로 도메인을 판별하여 이후 전략을 결정하는 패턴
"""
from __future__ import annotations

import os
import re
from typing import Any

import yaml

from structverify.core.schemas import SIRDocument
from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

CONFIDENCE_THRESHOLD = 0.6
DOMAIN_NAME_PATTERN = re.compile(r"^[a-z][a-z_]{0,29}$")

# 기본 시드 도메인 — 레지스트리 파일이 없을 때 초기값으로 사용
DEFAULT_SEED_DOMAINS: dict[str, str] = {
    "agriculture":  "농림수산식품 (농가, 경작면적, 수확량, 축산, 어업)",
    "economy":      "경제/경기 (GDP, 성장률, 소비, 수출입, 물가, 산업생산)",
    "finance":      "금융/증권 (금리, 환율, 주가, 대출, 가계부채, 보험)",
    "population":   "인구/가구 (출생, 사망, 혼인, 고령화, 인구구조)",
    "employment":   "고용/노동/임금 (취업률, 실업률, 임금, 근로시간)",
    "healthcare":   "보건/의료 (질병, 의료기관, 사망률, 건강보험)",
    "education":    "교육 (학생, 학교, 교육비, 진학률, 입시)",
    "policy":       "정책/행정 (예산, 법률, 복지, 지원금, 정부)",
    "environment":  "환경/에너지 (기후, 탄소, 재생에너지, 환경오염)",
    "general":      "분류 불가 또는 복합 도메인",
}

DOMAIN_CLASSIFY_PROMPT = """당신은 한국 통계/뉴스 도메인 분류 전문가입니다.
아래 문서를 읽고, 가장 적합한 도메인을 선택하거나 새로 생성하세요.

[현재 등록된 도메인 목록]
{domain_list}

[도메인 선택 규칙]
1. 위 목록에서 문서 내용과 가장 잘 맞는 도메인이 있으면 그 도메인을 선택하세요.
2. 목록에 적합한 도메인이 없을 때만 새 도메인을 만드세요.
   - 영어 소문자와 언더스코어(_)만 사용 (예: real_estate, it_industry)
   - 새 도메인 설명은 한국어로 간략히 작성
3. 분류가 모호하거나 복합 도메인이면 "general"을 선택하세요.

[예시]
문서: "통계청에 따르면 지난해 농가 인구는 216만 명으로 전년 대비 3.2% 감소했다."
→ {{"domain": "agriculture", "description": "농림수산식품 (농가, 경작면적, 수확량, 축산, 어업)", "is_new": false, "confidence": 0.95, "reason": "농가 인구 통계"}}

문서: "수도권 아파트 평균 매매가가 8억을 돌파하며 역대 최고치를 기록했다."
→ {{"domain": "real_estate", "description": "부동산 (아파트, 매매가, 전세, 분양)", "is_new": true, "confidence": 0.93, "reason": "부동산 가격 통계로 기존 목록에 없음"}}

[분류할 문서]
{text_preview}

JSON으로만 답하세요:
{{
  "domain": "도메인명",
  "description": "도메인 한국어 설명",
  "is_new": true 또는 false,
  "confidence": 0.0~1.0,
  "reason": "한 줄 근거"
}}"""


# ── DomainRegistry ────────────────────────────────────────────────────────

class DomainRegistry:
    """
    도메인 레지스트리 — {domain: description} 매핑을 파일로 영속 관리.

    registry.yaml 구조:
        agriculture: "농림수산식품 (농가, 경작면적, ...)"
        economy: "경제/경기 (GDP, 성장률, ...)"
        real_estate: "부동산 (아파트, 매매가, ...)"   ← 런타임에 추가됨

    사용법:
        registry = DomainRegistry("domain-packs/registry.yaml")
        domains = registry.load()           # {domain: description} 반환
        registry.register("real_estate", "부동산 관련 통계")
    """

    def __init__(self, registry_path: str = "domain-packs/registry.yaml"):
        self.registry_path = registry_path

    def load(self) -> dict[str, str]:
        """
        레지스트리 파일 로드.
        파일이 없으면 DEFAULT_SEED_DOMAINS를 파일로 저장 후 반환.
        """
        if not os.path.exists(self.registry_path):
            logger.info(f"레지스트리 없음 → 시드 도메인으로 초기화: {self.registry_path}")
            self._save(DEFAULT_SEED_DOMAINS)
            return dict(DEFAULT_SEED_DOMAINS)

        try:
            with open(self.registry_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return {k: str(v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"레지스트리 로드 실패 → 시드 사용: {e}")
            return dict(DEFAULT_SEED_DOMAINS)

    def register(self, domain: str, description: str) -> None:
        """
        새 도메인을 레지스트리에 추가하고 파일로 저장.
        이미 있으면 무시.
        """
        current = self.load()
        if domain in current:
            return

        current[domain] = description
        self._save(current)
        logger.info(f"새 도메인 등록: {domain} — {description}")

    def _save(self, data: dict[str, str]) -> None:
        os.makedirs(os.path.dirname(self.registry_path) or ".", exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=True)

    def format_for_prompt(self) -> str:
        """
        프롬프트 주입용 문자열 생성.
        예: "- agriculture: 농림수산식품 (농가, 경작면적, ...)"
        """
        domains = self.load()
        lines = [f"- {k}: {v}" for k, v in sorted(domains.items())]
        return "\n".join(lines)


# ── 메인 함수 ──────────────────────────────────────────────────────────────

async def classify_domain(
    sir_doc: SIRDocument,
    config: dict | None = None,
) -> tuple[str, str]:
    """
    SIR 문서의 도메인을 LLM으로 분류한다.

    반환값이 (domain, description) 튜플로 바뀐 이유:
      - domain만 반환하면 나중에 설명을 다시 조회해야 함
      - 한 번에 받아서 schema_inductor 등에서 도메인 힌트로 바로 활용 가능

    분류 로직:
      1) 레지스트리에서 기존 도메인 목록 + 설명 로드
      2) LLM 프롬프트에 목록 주입 → 기존 재사용 or 신규 생성
      3) 신규 도메인이면 레지스트리에 자동 저장
      4) confidence 낮으면 "general" fallback

    Args:
        sir_doc: 분류할 SIR 문서
        config: 설정 dict

    Returns:
        (domain, description) 튜플
        예: ("agriculture", "농림수산식품 (농가, 경작면적, ...)")
    """
    config = config or {}
    registry_path = config.get("domain_registry_path", "domain-packs/registry.yaml")
    registry = DomainRegistry(registry_path)

    preview = _build_text_preview(sir_doc)
    domain_list_str = registry.format_for_prompt()
    llm = LLMClient(config=config.get("llm", {}))

    try:
        result = await llm.generate_json(
            prompt=DOMAIN_CLASSIFY_PROMPT.format(
                domain_list=domain_list_str,
                text_preview=preview,
            ),
            system_prompt="도메인 분류 전문가. JSON으로만 답하세요.",
            model_tier="light",  # HCX-DASH-001
        )

        raw_domain    = result.get("domain", "general")
        description   = result.get("description", "")
        is_new        = bool(result.get("is_new", False))
        confidence    = float(result.get("confidence", 0.0))
        reason        = result.get("reason", "")

        # 도메인 형식 검증
        if not DOMAIN_NAME_PATTERN.match(raw_domain):
            logger.warning(f"도메인 형식 오류 '{raw_domain}' → general")
            raw_domain, description = "general", DEFAULT_SEED_DOMAINS["general"]

        # confidence 낮으면 general
        if confidence < CONFIDENCE_THRESHOLD:
            logger.warning(f"confidence 낮음 ({confidence:.2f}) → general")
            raw_domain, description = "general", DEFAULT_SEED_DOMAINS["general"]

        # 신규 도메인이면 레지스트리에 저장
        if is_new and raw_domain != "general":
            registry.register(raw_domain, description)

        # 기존 도메인이면 레지스트리의 공식 설명 사용 (LLM 설명이 다를 수 있음)
        if not is_new:
            registered = registry.load()
            description = registered.get(raw_domain, description)

        domain = raw_domain
        logger.info(
            f"도메인 분류: {domain} ({'신규' if is_new else '기존'}) "
            f"confidence={confidence:.2f}, reason={reason}"
        )

    except Exception as e:
        logger.error(f"도메인 분류 실패: {e}")
        domain, description = "general", DEFAULT_SEED_DOMAINS["general"]

    sir_doc.detected_domain = domain
    _load_domain_pack(domain, config)
    return domain, description


# ── 내부 유틸 ─────────────────────────────────────────────────────────────

def _build_text_preview(sir_doc: SIRDocument, max_chars: int = 600) -> str:
    """
    SIR 문서에서 분류에 유용한 미리보기 텍스트를 구성한다.
    heading 블록 우선, 이후 paragraph 추가. table/list 제외.
    """
    from structverify.core.schemas import BlockType

    parts: list[str] = []
    total = 0

    for block in sir_doc.blocks:
        if block.type == BlockType.HEADING and block.content:
            parts.append(block.content.strip())
            total += len(block.content)
            if total >= max_chars:
                break

    for block in sir_doc.blocks:
        if block.type == BlockType.PARAGRAPH and block.content:
            parts.append(block.content.strip())
            total += len(block.content)
            if total >= max_chars:
                break

    return " ".join(parts)[:max_chars]


def _load_domain_pack(domain: str, config: dict) -> dict[str, Any] | None:
    """
    domain-packs/{domain}/prompts.yaml 로드.
    없으면 None 반환 (에러 아님).
    """
    pack_dir = config.get("domain_packs_dir", "domain-packs")
    yaml_path = os.path.join(pack_dir, domain, "prompts.yaml")

    if not os.path.exists(yaml_path):
        logger.debug(f"Domain Pack 없음: {yaml_path}")
        return None

    try:
        with open(yaml_path, encoding="utf-8") as f:
            pack = yaml.safe_load(f)
        logger.info(f"Domain Pack 로드: {yaml_path}")
        return pack
    except Exception as e:
        logger.warning(f"Domain Pack 로드 실패: {yaml_path} — {e}")
        return None