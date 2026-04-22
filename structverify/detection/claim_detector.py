"""
detection/claim_detector.py — 검증 가능 주장 탐지 (Step 4)

[김예슬]
- check-worthiness 프롬프트 설계 및 튜닝 담당
- candidate scoring → LLM 2차 판별 구조 담당
- domain-packs 기반 도메인별 프롬프트 주입

[변경 요약]
- 기존: has_numeric=True 문장만 필터 → LLM check-worthiness
- 변경: LLM/학습 기반 sentence candidate scoring → 상위 후보만 LLM check-worthiness

[설계 원칙]
- Regex 필터(has_numeric 등) 로 1차 후보를 결정하지 않습니다.
- candidate_scorer.py의 Teacher LLM이 0~1 점수를 계산하고,
  threshold 이상인 문장만 이 check-worthiness 단계로 전달됩니다.
- 즉, Step 4를 다음 두 단계로 분리합니다:
  1) Sentence Candidate Detection (candidate_scorer.py — Teacher LLM)
  2) Claim Detection / Check-Worthiness (이 파일 — LLM 중량 모델)
"""
from __future__ import annotations

from structverify.core.schemas import Claim, ClaimType, SIRDocument, SourceOffset
from structverify.detection.candidate_scorer import score_candidate
from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

# TODO [김예슬]: 프롬프트 튜닝
#   - domain-packs/{domain}/prompts.yaml에서 도메인별 few-shot 예시 로드
#   - positive 예시: 공식 통계로 검증 가능한 수치 주장 2~3개
#   - negative 예시: 의견/감상/단순 이벤트 일정 2~3개
#   - claim_type 분류 기준 명확화 (increase/decrease/scale/comparison/forecast)
CHECK_WORTHY_PROMPT = """팩트체크 전문가로서 아래 문장이 공식 통계로 검증 가능한 수치 기반 주장인지 판별하세요.

기준:
1) 공식 통계 또는 공공 데이터와 연결 가능
2) 단순 일정/발언 소개/감상이 아니라 사실 주장
3) 가능한 경우 claim_type도 분류

문장: "{sentence}"

JSON: {{"is_check_worthy": true/false, "score": 0.0, "claim_type": "increase|decrease|scale|comparison|forecast|null"}}"""


async def detect_claims(
    sir_doc: SIRDocument,
    config: dict | None = None,
) -> list[Claim]:
    """
    SIR Tree에서 검증 가능한 주장 탐지.

    단계
    1) LLM 기반 sentence candidate scoring
    2) high-score 문장만 check-worthiness 판별 (LLM 중량 모델)
    3) threshold 이상 claim만 Claim 객체로 변환

    TODO [김예슬]: 도메인별 프롬프트 주입
      domain = sir_doc.detected_domain
      if domain:
          domain_examples = load_domain_prompts(domain)  # domain-packs 로드
          prompt = inject_few_shot(CHECK_WORTHY_PROMPT, domain_examples)

    TODO [김예슬]: claim_type 분류 정확도 개선
      - "increase": 증가/상승/올랐다
      - "decrease": 감소/하락/내렸다
      - "scale": 규모/비율/수준 언급
      - "comparison": A가 B보다 높다/낮다
      - "forecast": 전망/예상/목표
    """
    config = config or {}
    llm = LLMClient(config=config.get("llm", {}))

    cd_cfg = config.get("candidate_detection", {})
    candidate_threshold = float(cd_cfg.get("threshold", 0.65))
    min_conf = float(config.get("verification", {}).get("min_confidence", 0.7))

    candidate_count = 0
    claims: list[Claim] = []

    for block in sir_doc.blocks:
        for sent in block.sentences:
            # 1) LLM 기반 candidate scoring (LLM → heuristic fallback 순서)
            score, label, source, signals = await score_candidate(
                sentence=sent.text,
                config=config,
                context={
                    "block_id": block.block_id,
                    "domain": sir_doc.detected_domain,
                },
            )

            sent.candidate_score = score
            sent.candidate_label = label
            sent.candidate_source = source
            sent.candidate_signals = signals

            if score < candidate_threshold or not label:
                continue

            candidate_count += 1

            # 2) LLM check-worthiness (중량 모델 — 정밀 판별)
            cw_score, ctype = await _check_worthiness(llm, sent.text)
            if cw_score < min_conf:
                continue

            # 3) Claim 객체 생성
            claims.append(
                Claim(
                    doc_id=sir_doc.doc_id,
                    block_id=block.block_id,
                    sent_id=sent.sent_id,
                    claim_text=sent.text,
                    claim_type=ctype,
                    check_worthy_score=cw_score,
                    graph_anchor_id=sent.graph_anchor_id,
                    source_offset=SourceOffset(
                        char_start=sent.char_offset_start,
                        char_end=sent.char_offset_end,
                        page=block.source_offset.page,
                    ),
                )
            )

    logger.info(f"candidate 문장: {candidate_count}건")
    logger.info(f"검증 가능 주장: {len(claims)}건")
    return claims


async def _check_worthiness(
    llm: LLMClient,
    sentence: str,
) -> tuple[float, ClaimType | None]:
    """
    LLM 기반 check-worthiness 판별 (2차 정밀 판별).
    candidate detection 이후 상위 후보에만 적용.

    TODO [김예슬]: 오류 응답 처리 강화
      - JSON 파싱 실패 시 재시도 (최대 2회)
      - score 범위 검증 (0~1 클램핑)
    """
    try:
        r = await llm.generate_json(CHECK_WORTHY_PROMPT.format(sentence=sentence))
        is_check_worthy = bool(r.get("is_check_worthy", False))
        score = float(r.get("score", 0.0))
        if not is_check_worthy:
            return 0.0, None

        ct = r.get("claim_type")
        ctype = ClaimType(ct) if ct and ct != "null" else None
        return score, ctype
    except Exception as e:
        logger.error(f"check-worthiness 실패: {e}")
        return 0.0, None
