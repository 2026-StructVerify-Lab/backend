"""structverify.detection — Step 3~5 public API.

domain classify → claim detect → schema induce
"""
from structverify.detection.claim_detector import detect_claims
from structverify.detection.candidate_scorer import score_candidate
from structverify.detection.domain_classifier import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_SEED_DOMAINS,
    DOMAIN_NAME_PATTERN,
    DomainRegistry,
    classify_domain,
)
from structverify.detection.schema_inductor import induce_schemas, regenerate_schema

__all__ = [
    "CONFIDENCE_THRESHOLD",
    "DEFAULT_SEED_DOMAINS",
    "DOMAIN_NAME_PATTERN",
    "DomainRegistry",
    "classify_domain",
    "detect_claims",
    "induce_schemas",
    "regenerate_schema",
    "score_candidate",
]
