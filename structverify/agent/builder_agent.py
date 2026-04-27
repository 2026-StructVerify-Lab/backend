"""
agent/builder_agent.py — Agent B: 도메인 적응 관리자

[김예슬 - 2026-04-24]
pretrain_domain() 파이프라인 완성:
  1) KOSIS 메타 수집 (kosis_crawler — httpx 실제 호출)
  2) 합성 데이터 생성 (synthetic_generator — claim/schema + candidate detection)
  3) 학습 포맷 변환 (sample_builder — 5가지 태스크 포맷)
  4) NCP Tuning API 학습 (adapter_trainer — Object Storage + Tuning API + polling)
  5) 평가 → 합격 시 배포 (model.yaml 업데이트)

[경로 1] 사전학습 pretrain_domain():
  KOSIS 메타 → Self-Instruct 합성 → JSONL 변환 → NCP Object Storage
  → Tuning API → polling → 평가 → domain-packs/{domain}/model.yaml 배포

[경로 2] 피드백 학습 _trigger_adaptation():
  Human Review 수정 결과 → finetune 포맷 → 추가 NCP Tuning → 평가 → 배포
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
    Agent B: 도메인 적응 관리자.

    - pretrain_domain(): 서비스 전 사전학습 (1회)
    - log_feedback():    운영 피드백 기록
    - _trigger_adaptation(): 피드백 누적 시 추가 학습
    """

    def __init__(self, config: dict | None = None):
        self.config         = config or {}
        self.feedback_store = FeedbackStore(config=self.config)
        self.trainer        = AdapterTrainer(config=self.config)
        self.llm            = LLMClient(config=self.config.get("llm", {}))
        adapt_cfg           = self.config.get("adaptation", {})
        self.threshold      = int(adapt_cfg.get("feedback_threshold", 10))
        self.eval_min_score = float(adapt_cfg.get("eval_min_score", 0.85))

    async def pretrain_domain(
        self,
        domain: str,
        max_tables: int | None = None,
    ) -> str | None:
        """
        도메인 사전학습 전체 파이프라인 실행.

        Args:
            domain:     학습할 도메인 (예: "news", "agriculture")
            max_tables: 처리할 최대 KOSIS 통계표 수 (None=전체)

        Returns:
            배포된 adapter_path 또는 None (실패/평가 미통과)
        """
        logger.info(f"[Agent B] === 사전학습 시작: {domain} ===")

        # ── Step 0-1: KOSIS 메타데이터 수집 ──────────────────────────
        logger.info("[Agent B] Step 0-1: KOSIS 메타데이터 수집")
        catalog = await crawl_kosis_catalog(self.config)
        if not catalog:
            logger.error("[Agent B] 메타데이터 수집 실패 (KOSIS_API_KEY 확인 필요)")
            return None

        await save_to_db(catalog, self.config)
        logger.info(f"[Agent B] 메타데이터 {len(catalog)}건 수집 완료")

        # ── Step 0-2: 합성 학습 데이터 생성 ──────────────────────────
        logger.info("[Agent B] Step 0-2: Self-Instruct 합성 데이터 생성")
        synthetic = await generate_synthetic_pairs(
            catalog=catalog,
            llm=self.llm,
            claims_per_table=3,
            max_tables=max_tables,
        )
        if not synthetic:
            logger.error("[Agent B] 합성 데이터 생성 실패")
            return None

        # JSONL 파일로 저장 (ml/data/)
        await save_synthetic_data(synthetic)
        logger.info(f"[Agent B] 합성 데이터 {len(synthetic)}쌍 생성 완료")

        # ── Step 0-3: 학습 포맷 변환 ─────────────────────────────────
        logger.info("[Agent B] Step 0-3: 학습 포맷 변환 (pretrain)")
        samples = build_training_samples(synthetic=synthetic, mode="pretrain")
        logger.info(f"[Agent B] 학습 샘플 {len(samples)}건 생성")

        if not samples:
            logger.error("[Agent B] 학습 샘플 없음")
            return None

        # ── Step 0-4: NCP Tuning API 학습 ────────────────────────────
        logger.info("[Agent B] Step 0-4: NCP Tuning API 학습 시작")
        adapter_path = await self.trainer.train(domain, samples)
        if not adapter_path:
            logger.error("[Agent B] 학습 실패 (NCP Tuning API 확인 필요)")
            return None

        # ── Step 0-5: 평가 → 배포 ────────────────────────────────────
        benchmark = f"domain-packs/{domain}/eval_dataset.jsonl"
        score = await self.trainer.evaluate(adapter_path, benchmark)
        logger.info(f"[Agent B] 평가 점수: {score:.3f} (기준: {self.eval_min_score})")

        if score < self.eval_min_score:
            logger.warning(
                f"[Agent B] 평가 미통과 ({score:.3f} < {self.eval_min_score}) "
                f"— Adapter 배포 안 함"
            )
            return None

        deployed = await self.trainer.deploy(adapter_path, domain)
        if deployed:
            logger.info(f"[Agent B] === 사전학습 완료: {domain} Adapter 배포 ===")
        return adapter_path

    # ── 피드백 루프 ──────────────────────────────────────────────────────

    async def log_feedback(self, event: FeedbackEvent) -> None:
        """
        Step 11: Human Review 피드백 저장 + 임계값 도달 시 학습 트리거.

        TODO [박재윤]: feedback_store.save() → feedback_events 테이블 INSERT
        """
        await self.feedback_store.save(event)
        logger.info(f"[Agent B] 피드백 기록: {event.feedback_type.value}")

        count = await self.feedback_store.count_by_domain()
        logger.debug(f"[Agent B] 누적 피드백: {count}건 (임계값: {self.threshold})")

        if count >= self.threshold:
            logger.info(f"[Agent B] 임계값 도달 → Adaptation 트리거")
            await self._trigger_adaptation()

    async def _trigger_adaptation(self) -> None:
        """
        Step 12: 피드백 기반 추가 학습.

        pending 피드백 수집 → finetune 포맷 변환 → NCP Tuning → 평가 → 배포.
        """
        logger.info("[Agent B] 피드백 Adaptation 시작")

        events = await self.feedback_store.get_pending()
        if not events:
            logger.warning("[Agent B] pending 피드백 없음")
            return

        samples = build_training_samples(events=events, mode="finetune")
        logger.info(f"[Agent B] 피드백 샘플 {len(samples)}건")

        if not samples:
            return

        adapter_path = await self.trainer.train("news", samples)
        if not adapter_path:
            return

        benchmark = "domain-packs/news/eval_dataset.jsonl"
        score = await self.trainer.evaluate(adapter_path, benchmark)

        if score >= self.eval_min_score:
            await self.trainer.deploy(adapter_path, "news")
            logger.info("[Agent B] 피드백 Adapter 배포 완료")
        else:
            logger.warning(f"[Agent B] 피드백 Adapter 평가 미통과: {score:.3f}")

    async def generate_domain_pack(self, domain: str) -> DomainPack:
        """
        새 도메인용 Domain Pack 디렉토리 및 기본 파일 생성.

        TODO [김예슬]: KOSIS 메타 기반 few-shot 예시 자동 생성
        """
        import os, yaml

        pack_dir = os.path.join("domain-packs", domain)
        os.makedirs(pack_dir, exist_ok=True)

        prompts_yaml = os.path.join(pack_dir, "prompts.yaml")
        if not os.path.exists(prompts_yaml):
            with open(prompts_yaml, "w", encoding="utf-8") as f:
                yaml.dump(
                    {"domain": domain, "few_shot_examples": []},
                    f, allow_unicode=True,
                )

        logger.info(f"[Agent B] Domain Pack 생성: {pack_dir}")
        return DomainPack(pack_id=f"pack_{domain}_v1", domain=domain, version="1.0")