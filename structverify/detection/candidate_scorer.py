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

from typing import Any

from structverify.detection._config import candidate_detection_config
from structverify.detection.candidate.heuristic import _score_candidate_heuristic
from structverify.detection.candidate.llm import _score_candidate_llm
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


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
    cd_cfg = candidate_detection_config(config)
    use_llm = cd_cfg.get("teacher_llm_fallback", True)
    threshold = float(cd_cfg.get("threshold", 0.65))
    domain = (context or {}).get("domain")

    if use_llm:
        try:
            return await _score_candidate_llm(
                sentence,
                config=config,
                threshold=threshold,
                domain=domain,
            )
        except Exception as e:
            logger.warning(f"candidate LLM 판별 실패 — heuristic fallback 사용: {e}")

    # fallback: LLM 실패 시만 사용
    return _score_candidate_heuristic(sentence, threshold=threshold)
