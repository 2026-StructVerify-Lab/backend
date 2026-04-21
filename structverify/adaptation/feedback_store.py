"""
adaptation/feedback_store.py — Feedback Store (Human Review / 실패 사례)

[참고] Feedback Adaptation for RAG (arXiv 2604.06647)
  피드백 수집 → 학습 데이터 변환 → 모델 개선 비동기 루프
"""
from __future__ import annotations
from structverify.core.schemas import FeedbackEvent
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


class FeedbackStore:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._events: list[FeedbackEvent] = []  # TODO: PostgreSQL로 교체

    async def save(self, event: FeedbackEvent) -> None:
        """피드백 이벤트 저장 — TODO: DB INSERT 구현"""
        self._events.append(event)

    async def count_by_domain(self) -> int:
        """도메인별 미처리 피드백 수 — TODO: DB COUNT 쿼리"""
        return len(self._events)

    async def get_pending(self) -> list[FeedbackEvent]:
        """미처리 피드백 조회 — TODO: DB SELECT WHERE status=pending"""
        pending = list(self._events)
        self._events.clear()
        return pending
