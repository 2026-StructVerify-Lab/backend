"""
agent/builder_agent.py — Agent B: 도메인 적응 관리자

[김예슬]
- KOSIS 메타데이터 기반 Self-Instruct 합성 데이터 생성 및 LoRA 학습 루프 담당
- 피드백 기반 추가 학습(_trigger_adaptation) 담당
- synthetic_generator, sample_builder, adapter_trainer 연결 오케스트레이션

[학습 파이프라인]
  [경로 1] 사전학습 (pretrain_domain):
    KOSIS 메타 → LLM Self-Instruct → positive/negative 쌍 생성
    → 학습 포맷 변환 → LoRA Fine-tuning → 평가 → 배포

  [경로 2] 피드백 학습 (_trigger_adaptation):
    Human Review 수정 결과 → 추가 LoRA Fine-tuning → 평가 → 배포

[변경 요약]
- synthetic_generator가 sentence_to_candidate 태스크를 함께 생성하도록 확장
- build_training_samples(mode="pretrain")만 호출하면 candidate detection 학습 데이터까지 포함
"""
from __future__ import annotations

from structverify.core.schemas import DomainPack, FeedbackEvent
from structverify.adaptation.feedback_store import FeedbackStore
from structverify.adaptation.kosis_crawler import crawl_kosis_catalog, save_to_db
from structverify.adaptation.sample_builder import build_training_samples
from structverify.adaptation.synthetic_generator import generate_synthetic_pairs, save_synthetic_data
from structverify.adaptation.adapter_trainer import AdapterTrainer
from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


class BuilderAgent:
    """
    Agent B: 도메인 적응 관리자
    - pretrain_domain(): 서비스 전 사전학습
    - log_feedback(): 운영 피드백 기록
    - _trigger_adaptation(): 피드백 누적 시 추가 학습
    """
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.feedback_store = FeedbackStore(config=self.config)
        self.trainer = AdapterTrainer(config=self.config)
        self.llm = LLMClient(config=self.config.get("llm", {}))
        self.threshold = self.config.get("adaptation", {}).get("feedback_threshold", 10)
        self.eval_min_score = self.config.get("adaptation", {}).get("eval_min_score", 0.85)

    async def pretrain_domain(self, domain: str, max_tables: int | None = None) -> str | None:
        """
        서비스 시작 전 도메인 사전학습 수행.

        흐름
        1) KOSIS 메타데이터 수집
        2) synthetic positive/negative candidate + claim/schema 샘플 생성
        3) 학습 포맷 변환
        4) LoRA 학습 → 평가 → 배포

        TODO [김예슬]: synthetic_generator.py 학습 샘플 생성 품질 검증
          - positive 샘플: KOSIS 통계 수치를 정확히 인용한 문장
          - negative 샘플: 수치를 변형/맥락을 바꾼 문장
          - candidate detection 샘플: 검증 가능/불가능 문장 쌍

        TODO [김예슬]: sample_builder.py pretrain 포맷 검증
          - HuggingFace Dataset 형식으로 변환
          - instruction/input/output 필드 구조 확인

        TODO [김예슬]: adapter_trainer.py LoRA 학습 구현
          - HuggingFace PEFT (LoRA) 라이브러리 사용
          - MLflow로 실험 추적 (mlflow_tracking_uri 설정)
        """
        logger.info(f"[Agent B] === 사전학습 시작: {domain} ===")

        logger.info("[Agent B] Step 0-1: KOSIS 메타데이터 수집")
        catalog = await crawl_kosis_catalog(self.config)
        if not catalog:
            logger.error("[Agent B] 메타데이터 수집 실패 — 사전학습 중단")
            return None

        # TODO [박재윤]: save_to_db가 실제 DB에 저장하는지 확인
        await save_to_db(catalog, self.config)
        logger.info(f"[Agent B] 메타데이터 {len(catalog)}건 수집 + DB 저장 완료")

        logger.info("[Agent B] Step 0-2: 합성 학습 데이터 생성")
        synthetic = await generate_synthetic_pairs(
            catalog=catalog,
            llm=self.llm,
            claims_per_table=3,
            max_tables=max_tables,
        )
        if not synthetic:
            logger.error("[Agent B] 합성 데이터 생성 실패 — 사전학습 중단")
            return None

        await save_synthetic_data(synthetic)

        logger.info("[Agent B] Step 0-3: 학습 포맷 변환")
        # mode="pretrain": claim/schema + candidate detection 샘플 모두 포함
        samples = build_training_samples(synthetic=synthetic, mode="pretrain")
        logger.info(f"[Agent B] 학습 샘플 {len(samples)}건 생성")

        logger.info("[Agent B] Step 0-4: LoRA 학습")
        # TODO [김예슬]: adapter_trainer.train() 실제 PEFT LoRA 학습 구현
        adapter_path = await self.trainer.train(domain, samples)

        if not adapter_path:
            logger.error("[Agent B] 학습 실패")
            return None

        score = await self.trainer.evaluate(
            adapter_path,
            benchmark=f"domain-packs/{domain}/eval_dataset.json",
        )
        logger.info(f"[Agent B] 평가 점수: {score:.3f} (기준: {self.eval_min_score})")

        if score < self.eval_min_score:
            logger.warning("[Agent B] 평가 미통과 — Adapter 배포 안 함")
            return None

        await self.trainer.deploy(adapter_path, domain)
        logger.info(f"[Agent B] === 사전학습 완료: {domain} Adapter 배포됨 ===")
        return adapter_path

    async def log_feedback(self, event: FeedbackEvent) -> None:
        """
        Step 11: 피드백 저장

        TODO [김예슬]: feedback_store.save() 실제 DB 저장 구현
        TODO [박재윤]: FeedbackEvent → feedback_events 테이블 INSERT 확인
        """
        await self.feedback_store.save(event)
        logger.info(f"[Agent B] 피드백 기록: {event.feedback_type.value}")

        count = await self.feedback_store.count_by_domain()
        if count >= self.threshold:
            await self._trigger_adaptation()

    async def _trigger_adaptation(self) -> None:
        """
        Step 12: 피드백 기반 Adaptation

        TODO [김예슬]: 피드백 데이터로 sample_builder.py (mode="finetune") 포맷 변환 검증
          - Human Review 수정 결과: corrected_verdict, reviewer_note
          - 학습 샘플: (claim_text, original_verdict) → corrected_verdict
          - instruction tuning 형식으로 변환

        TODO [김예슬]: 피드백 학습 후 기존 adapter와 merge 전략 결정
          - 완전 교체 vs LoRA weight merge
        """
        logger.info("[Agent B] 피드백 Adaptation 트리거됨")
        events = await self.feedback_store.get_pending()
        samples = build_training_samples(events=events, mode="finetune")
        logger.info(f"[Agent B] 피드백 샘플 {len(samples)}건 생성")

        adapter_path = await self.trainer.train("news", samples)
        if adapter_path:
            score = await self.trainer.evaluate(adapter_path, benchmark="domain-packs/news/eval_dataset.json")
            if score >= self.eval_min_score:
                await self.trainer.deploy(adapter_path, "news")
                logger.info("[Agent B] 피드백 Adapter 배포 완료")

    async def generate_domain_pack(self, domain: str) -> DomainPack:
        """
        새 도메인용 Domain Pack 생성 placeholder

        TODO [김예슬]: domain-packs/{domain}/prompts.yaml 자동 생성
          - KOSIS 메타데이터 기반 few-shot 예시 자동 생성
          - domain-packs 디렉토리 구조 생성
        """
        logger.info(f"[Agent B] Domain Pack 생성: {domain}")
        return DomainPack(pack_id=f"pack_{domain}_v1", domain=domain, version="1.0")
