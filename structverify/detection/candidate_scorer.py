"""
detection/candidate_scorer.py — 문장 단위 검증 후보 점수화

[김예슬]
- Teacher LLM 기반 0~1 점수화 로직 담당
- heuristic fallback은 운영 안정성을 위한 보조 수단
- 학습 데이터 충분 누적 후 소형 분류 모델(LoRA fine-tuned)로 교체 계획

[설계 원칙]
- regex/rule만으로 후보를 결정하지 않는다.
- surface signal + teacher LLM + weak supervision 규칙을 결합할 수 있는 인터페이스 제공.
- 현재 버전: "teacher LLM + heuristic fallback" 구조.
- 이후 작은 classifier를 붙일 때 이 파일만 교체하면 된다.

[LLM 학습 계획]
  Phase 1: Teacher LLM (HCX-DASH-001)이 직접 판단 → 결과를 학습 샘플로 저장
  Phase 2: Step 0 합성 데이터 + 운영 피드백 누적 → LoRA fine-tuning
  Phase 3: 학습된 경량 모델로 교체 (비용 절감 + 속도 향상)

출력
- candidate_score: 0~1
- candidate_label: bool
- candidate_source: 점수 출처
- candidate_signals: 분석용 signal
"""
from __future__ import annotations

import re
from typing import Any

from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

# TODO [김예슬]: 프롬프트 튜닝 — domain-packs의 few-shot 예시 주입
#   - 도메인별 positive/negative 예시 2~3개씩 추가
#   - "공식 통계와 연결 가능" 기준을 예시로 명확히 제시
CANDIDATE_PROMPT = """당신은 수치 기반 팩트체크 시스템의 1차 후보 탐지기입니다.
아래 문장이 "공식 통계나 구조화된 데이터로 검증할 만한 후보 문장"인지 판단하세요.

판단 기준:
1. 수치/비율/규모/시점/대상 중 일부가 드러나는가?
2. 의견/감상/단순 이벤트 일정이 아니라 검증 가능한 사실 주장인가?
3. 공식 통계 또는 공공 데이터와 연결될 가능성이 있는가?

문장: "{sentence}"

중요:
- candidate_label이 true이면 candidate_score는 반드시 0.5 이상이어야 합니다.
- candidate_label이 false이면 candidate_score는 반드시 0.5 미만이어야 합니다.
- JSON 앞뒤에 설명 문장을 절대 붙이지 마세요.

JSON으로만 답하세요:
{{
  "candidate_score": 0.0,
  "candidate_label": false,
  "reason": "짧은 근거",
  "signals": {{
    "has_quantity": false,
    "has_time_expr": false,
    "has_population": false,
    "has_comparison_expr": false
  }}
}}
"""

# ── heuristic fallback 패턴 (LLM 실패 시만 사용) ────────────────────────
# 아래 패턴들은 LLM이 호출 불가능할 때만 사용하는 fallback입니다.
# 운영 환경에서는 LLM 판단이 우선입니다.
TIME_PATTERN = re.compile(r"\d{4}년|\d+월|\d+분기|전년|지난해|올해")
COMPARISON_PATTERN = re.compile(r"증가|감소|상승|하락|올랐다|내렸다|대비|비율|점유율|이상|이하|안팎")
POPULATION_PATTERN = re.compile(r"국내|전국|가구|가계|농가|학생|청년|고령자|근로자|기업|미국|일본|유럽|한국")
NUMBER_PATTERN = re.compile(r"\d")


async def score_candidate(
    sentence: str,
    config: dict | None = None,
    context: dict[str, Any] | None = None,
) -> tuple[float, bool, str, dict[str, Any]]:
    """
    문장 후보 점수 계산.

    현재 로직
    1) teacher LLM 시도 (HCX-DASH-001 경량 모델)
    2) 실패 시 heuristic fallback

    TODO [김예슬]: 도메인 컨텍스트 활용
      - context["domain"]을 프롬프트에 주입하여 도메인별 판단 기준 적용
      - domain-packs/{domain}/prompts.yaml의 candidate 예시 주입

    TODO [김예슬]: 학습 데이터 수집 로직 추가
      - teacher LLM 판단 결과를 DB에 저장 (sample_builder.py 연동)
      - 나중에 LoRA fine-tuning에 활용

    TODO [김예슬]: 소형 분류 모델 교체 로직 (Phase 3)
      - 학습된 adapter 경로 확인 → 있으면 PEFT 모델 추론
      - adapter_path = config.get("adaptation", {}).get("adapter_path")
      - if adapter_path: return _score_with_trained_model(sentence, adapter_path)
    """
    config = config or {}
    cd_cfg = config.get("candidate_detection", {})
    use_llm = cd_cfg.get("teacher_llm_fallback", True)
    threshold = float(cd_cfg.get("threshold", 0.65))

    if use_llm:
        try:
            llm = LLMClient(config=config.get("llm", {}))
            result = await llm.generate_json(
                prompt=CANDIDATE_PROMPT.format(sentence=sentence),
                system_prompt="팩트체크 candidate detector. JSON으로만 답하세요.",
            )
            # score = float(result.get("candidate_score", 0.0))
            # label = bool(result.get("candidate_label", score >= threshold))
            # signals = result.get("signals", {}) or {}
            # signals["reason"] = result.get("reason")
            # return score, label, "teacher_llm", signals
            score = float(result.get("candidate_score", 0.0) or 0.0)
            label = bool(result.get("candidate_label", score >= threshold))

            # LLM이 label=true인데 score를 0으로 주는 경우 방어
            if label and score < threshold:
                score = max(score, 0.75)

            signals = result.get("signals", {}) or {}
            signals["reason"] = result.get("reason")
            signals["raw_llm_result"] = result

            return score, label, "teacher_llm", signals
        except Exception as e:
            logger.warning(f"candidate LLM 판별 실패 — heuristic fallback 사용: {e}")

    # fallback: LLM 실패 시만 사용
    return _score_candidate_heuristic(sentence, threshold=threshold)


def _score_candidate_heuristic(
    sentence: str,
    threshold: float = 0.65,
) -> tuple[float, bool, str, dict[str, Any]]:
    """
    최소한의 fallback heuristic.

    TODO [김예슬]: 논문 실험 baseline으로도 활용 가능
      - 이 함수의 성능(F1, precision, recall)을 측정하고
        teacher LLM 및 fine-tuned 모델과 비교

    주의: 이 heuristic은 LLM 호출 실패 시만 사용합니다.
    Rule 기반으로 검증 후보를 결정하는 용도로 사용하지 마세요.
    """
    has_quantity = bool(NUMBER_PATTERN.search(sentence))
    has_time_expr = bool(TIME_PATTERN.search(sentence))
    has_population = bool(POPULATION_PATTERN.search(sentence))
    has_comparison_expr = bool(COMPARISON_PATTERN.search(sentence))

    score = 0.0
    if has_quantity:
        score += 0.35
    if has_time_expr:
        score += 0.20
    if has_population:
        score += 0.20
    if has_comparison_expr:
        score += 0.25

    score = min(score, 1.0)
    label = score >= threshold

    signals = {
        "has_quantity": has_quantity,
        "has_time_expr": has_time_expr,
        "has_population": has_population,
        "has_comparison_expr": has_comparison_expr,
        "reason": "heuristic_fallback",
    }
    return score, label, "heuristic_fallback", signals