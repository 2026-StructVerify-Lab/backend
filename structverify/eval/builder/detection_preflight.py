"""Production claim-detection gate before freezing eval articles."""
from __future__ import annotations

from structverify.core.schemas import SourceType
from structverify.detection.claim_detector import detect_claims
from structverify.detection.domain_classifier import classify_domain
from structverify.eval.builder.text_utils import normalize_claim_text
from structverify.preprocessing.sir_builder import build_sir
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


def _count_gold_matches(
    gold_claim_texts: list[str],
    pipeline_claim_texts: list[str],
) -> int:
    gold_norm = [normalize_claim_text(g) for g in gold_claim_texts if g.strip()]
    pred_norm = [normalize_claim_text(p) for p in pipeline_claim_texts if p.strip()]
    matched = 0
    used: set[int] = set()
    for g in gold_norm:
        if not g:
            continue
        for i, p in enumerate(pred_norm):
            if i in used:
                continue
            if g == p or g in p or p in g:
                matched += 1
                used.add(i)
                break
    return matched


async def validate_article_detection(
    article_text: str,
    gold_claim_texts: list[str],
    config: dict,
    *,
    min_claims: int = 1,
    min_gold_matches: int = 1,
) -> tuple[bool, list[str]]:
    """
    Run production Step 3–4 (domain classify + detect_claims) on article_text.

    Uses *production* config as passed in (no harness threshold overrides).
    """
    errors: list[str] = []
    sir_doc = build_sir(article_text.strip(), SourceType.TEXT)
    if not sir_doc.blocks:
        return False, ["detection_preflight: empty SIR document"]

    domain, _desc = await classify_domain(sir_doc, config)
    config = {**config, "detected_domain": domain}

    try:
        claims = await detect_claims(sir_doc, config)
    except Exception as e:
        logger.warning(f"detection_preflight failed: {e}")
        return False, [f"detection_preflight: detect_claims error: {e}"]

    n = len(claims)
    if n < min_claims:
        errors.append(
            f"detection_preflight: pipeline detected {n} claims "
            f"(min {min_claims})"
        )

    pipeline_texts = [c.claim_text for c in claims]
    gold_matched = _count_gold_matches(gold_claim_texts, pipeline_texts)
    if gold_matched < min_gold_matches:
        errors.append(
            f"detection_preflight: {gold_matched} gold claim(s) matched "
            f"detected text (min {min_gold_matches})"
        )

    if errors:
        logger.info(
            f"detection_preflight reject domain={domain} "
            f"detected={n} gold_matches={gold_matched}"
        )
    return len(errors) == 0, errors
