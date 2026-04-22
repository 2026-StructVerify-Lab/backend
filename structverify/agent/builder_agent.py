"""
agent/builder_agent.py — Domain Adaptation Agent (Agent B)

두 가지 학습 경로를 관리한다:
  - 사전학습 (pretrain): 서비스 전, KOSIS 메타 → 합성 데이터 → LoRA 학습 (Step 0)
  - 피드백 학습 (finetune): 서비스 후, Human Review → 추가 학습 (Step 11~12)

[참고] Self-Instruct (Wang et al., ACL 2023)
  - https://github.com/yizhongw/self-instruct
  - KOSIS 메타데이터를 seed로 합성 학습 데이터를 자동 생성하는 사전학습 라인

[참고] Feedback Adaptation for RAG (arXiv 2604.06647)
  운영 피드백을 비동기 학습에 활용하는 루프 설계

[참고] KnowLA: KG-aware PEFT (NAACL 2024)
  지식 그래프 정보를 PEFT adapter에 반영하는 도메인 적응형 학습 전략
"""
from __future__ import annotations
from structverify.core.schemas import FeedbackEvent, DomainPack
from structverify.adaptation.feedback_store import FeedbackStore
from structverify.adaptation.sample_builder import build_training_samples
from structverify.adaptation.kosis_crawler import crawl_kosis_catalog, save_to_db
from structverify.adaptation.synthetic_generator import generate_synthetic_pairs, save_synthetic_data
from structverify.adaptation.adapter_trainer import AdapterTrainer
from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


class BuilderAgent:
    """
    Agent B: 도메인 적응 관리자.
    - pretrain_domain(): 서비스 전 사전학습 (Step 0)
    - log_feedback() + _trigger_adaptation(): 서비스 후 피드백 학습 (Step 11~12)
    """
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.feedback_store = FeedbackStore(config=self.config)
        self.trainer = AdapterTrainer(config=self.config)
        self.llm = LLMClient(config=self.config.get("llm", {}))
        self.threshold = self.config.get("adaptation", {}).get("feedback_threshold", 10)
        self.eval_min_score = self.config.get("adaptation", {}).get("eval_min_score", 0.85)

    # ═══════════════════════════════════════
    # Step 0: 사전학습 (서비스 시작 전 1회)
    # ═══════════════════════════════════════

    async def pretrain_domain(self, domain: str, max_tables: int | None = None) -> str | None:
        """
        서비스 시작 전 도메인 사전학습을 수행한다.

        Self-Instruct 방식으로 KOSIS 메타데이터 → 합성 학습 데이터 → LoRA 학습.

        흐름:
          Step 0-1: KOSIS 메타데이터 수집 (+ RAG용 임베딩 저장)
          Step 0-2: 합성 학습 데이터 생성 (LLM이 자동 생성)
          Step 0-3: 학습 포맷 변환
          Step 0-4: LoRA 학습 → 평가 → 배포

        Args:
            domain: 도메인 이름 (e.g., "news")
            max_tables: 처리할 최대 통계표 수 (None이면 전체)

        Returns:
            배포된 adapter 경로 또는 None (평가 실패 시)
        """
        logger.info(f"[Agent B] === 사전학습 시작: {domain} ===")

        # Step 0-1: KOSIS 메타데이터 수집
        logger.info("[Agent B] Step 0-1: KOSIS 메타데이터 수집")
        catalog = await crawl_kosis_catalog(self.config)
        if not catalog:
            logger.error("[Agent B] 메타데이터 수집 실패 — 사전학습 중단")
            return None

        # 수집 결과를 DB에 저장 (RAG용 임베딩 포함)
        await save_to_db(catalog, self.config)
        logger.info(f"[Agent B] 메타데이터 {len(catalog)}건 수집 + DB 저장 완료")

        # Step 0-2: 합성 학습 데이터 생성
        logger.info("[Agent B] Step 0-2: 합성 학습 데이터 생성 (Self-Instruct)")
        synthetic = await generate_synthetic_pairs(
            catalog, self.llm,
            claims_per_table=3,
            max_tables=max_tables,
        )
        if not synthetic:
            logger.error("[Agent B] 합성 데이터 생성 실패 — 사전학습 중단")
            return None

        # 합성 데이터 파일로 백업 저장
        await save_synthetic_data(synthetic)

        # Step 0-3: 학습 포맷 변환
        logger.info("[Agent B] Step 0-3: 학습 포맷 변환")
        samples = build_training_samples(synthetic=synthetic, mode="pretrain")
        logger.info(f"[Agent B] 학습 샘플 {len(samples)}건 생성")

        # Step 0-4: LoRA 학습 → 평가 → 배포
        logger.info("[Agent B] Step 0-4: LoRA 학습")
        adapter_path = await self.trainer.train(domain, samples)

        if adapter_path:
            score = await self.trainer.evaluate(adapter_path, benchmark=f"domain-packs/{domain}/eval_dataset.json")
            logger.info(f"[Agent B] 평가 점수: {score:.3f} (기준: {self.eval_min_score})")

            if score >= self.eval_min_score:
                await self.trainer.deploy(adapter_path, domain)
                logger.info(f"[Agent B] === 사전학습 완료: {domain} Adapter 배포됨 ===")
                return adapter_path
            else:
                logger.warning(f"[Agent B] 평가 미통과 — Adapter 배포 안 함")
                return None
        else:
            logger.error("[Agent B] 학습 실패")
            return None

    # ═══════════════════════════════════════
    # Step 11~12: 피드백 기반 학습 (서비스 후)
    # ═══════════════════════════════════════

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
        Step 12: 피드백 기반 Adaptation 트리거
        기존 사전학습 Adapter 위에 추가 학습 (continual learning)
        """
        logger.info("[Agent B] 피드백 Adaptation 트리거됨")
        events = await self.feedback_store.get_pending()

        # 학습 샘플 생성 (mode="finetune")
        samples = build_training_samples(events=events, mode="finetune")
        logger.info(f"[Agent B] 피드백 샘플 {len(samples)}건 생성")

        # LoRA 추가 학습 (사전학습 Adapter 위에)
        adapter_path = await self.trainer.train("news", samples)
        if adapter_path:
            score = await self.trainer.evaluate(adapter_path, benchmark="domain-packs/news/eval_dataset.json")
            if score >= self.eval_min_score:
                await self.trainer.deploy(adapter_path, "news")
                logger.info("[Agent B] 피드백 Adapter 배포 완료")

    # ═══════════════════════════════════════
    # Domain Pack 관리
    # ═══════════════════════════════════════

    async def generate_domain_pack(self, domain: str) -> DomainPack:
        """
        새 도메인에 대한 Domain Pack을 자동 생성한다.
        TODO: LLM 기반 Schema Template + Graph Ontology 자동 생성
        TODO: Connector Mapping Policy 자동 설정
        """
        logger.info(f"[Agent B] Domain Pack 생성: {domain}")
        return DomainPack(pack_id=f"pack_{domain}_v1", domain=domain, version="1.0")
