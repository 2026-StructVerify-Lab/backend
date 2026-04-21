"""
agent/builder_agent.py — Domain Adaptation Agent (Agent B)

비동기 설계·학습 담당. 운영 중 피드백을 수집하여 도메인별 Adapter를 개선한다.

[참고] Feedback Adaptation for RAG (arXiv 2604.06647)
  운영 피드백을 비동기 학습에 활용하는 루프 설계

[참고] KnowLA: KG-aware PEFT (NAACL 2024)
  지식 그래프 정보를 PEFT adapter에 반영하는 도메인 적응형 학습 전략
"""
from __future__ import annotations
from structverify.core.schemas import FeedbackEvent, DomainPack
from structverify.adaptation.feedback_store import FeedbackStore
from structverify.adaptation.sample_builder import build_training_samples
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


class BuilderAgent:
    """
    Agent B: 비동기 도메인 적응.
    Step 11~12를 처리하며 Adapter 재학습을 관리한다.
    """
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.feedback_store = FeedbackStore(config=self.config)
        self.threshold = self.config.get("adaptation", {}).get("feedback_threshold", 10)

    async def log_feedback(self, event: FeedbackEvent) -> None:
        """Step 11: 피드백 기록"""
        await self.feedback_store.save(event)
        logger.info(f"[Agent B] 피드백 기록: {event.feedback_type.value}")

        # 피드백 임계치 도달 시 학습 트리거
        count = await self.feedback_store.count_by_domain()
        if count >= self.threshold:
            await self._trigger_adaptation()

    async def _trigger_adaptation(self) -> None:
        """
        Step 12: Domain Adaptation 트리거
        1) 피드백 수집 → 2) 학습 샘플 생성 → 3) Adapter 재학습 → 4) 평가 → 5) 배포
        """
        logger.info("[Agent B] Adaptation 트리거됨")
        events = await self.feedback_store.get_pending()

        # 학습 샘플 생성
        samples = build_training_samples(events)
        logger.info(f"[Agent B] 학습 샘플 {len(samples)}건 생성")

        # TODO: PEFT/LoRA Adapter 학습 트리거
        # TODO: Evaluation Benchmark 실행
        # TODO: 통과 시 Domain Pack 버전 갱신 (MLflow)
        # TODO: Runtime Agent에 새 Adapter 핫스왑

    async def generate_domain_pack(self, domain: str) -> DomainPack:
        """
        새 도메인에 대한 Domain Pack을 자동 생성한다.
        TODO: LLM 기반 Schema Template + Graph Ontology 자동 생성
        TODO: Connector Mapping Policy 자동 설정
        """
        logger.info(f"[Agent B] Domain Pack 생성: {domain}")
        return DomainPack(pack_id=f"pack_{domain}_v1", domain=domain, version="1.0")
