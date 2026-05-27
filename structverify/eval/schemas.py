"""Pydantic models for 3-axis eval datasets and run artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

VerdictType = Literal["match", "mismatch", "unverifiable"]
CaseType = Literal["atomic", "context", "bundle"]
LabelMethod = Literal["kosis_probe", "value_perturbation"]


class OutcomeCase(BaseModel):
    case_id: str
    case_type: CaseType = "atomic"
    claim_text: str
    expected_verdict: VerdictType
    indicator: str | None = None
    time_period: str | None = None
    unit: str | None = None
    stated_value: float | None = None
    official_value: float | None = None
    domain: str | None = None
    context_text: str | None = None
    reference_stat_id: str | None = None
    kosis_org_id: str | None = None
    label_method: LabelMethod = "kosis_probe"


class OutcomeManifest(BaseModel):
    dataset_id: str
    status: Literal["building", "frozen"] = "building"
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    case_count: int = 0
    match_count: int = 0
    mismatch_count: int = 0
    builder_config_hash: str | None = None
    claims_sha256: str | None = None


class ComponentDetectionRow(BaseModel):
    row_id: str
    text: str
    should_extract: bool
    source_case_id: str | None = None


class ComponentSchemaRow(BaseModel):
    row_id: str
    claim_text: str
    domain: str
    context_text: str | None = None
    expected_indicator: str | None = None
    expected_value: float | None = None
    expected_time_period: str | None = None
    expected_unit: str | None = None
    source_case_id: str | None = None


class ComponentRetrievalRow(BaseModel):
    row_id: str
    keyword: str
    indicator: str | None = None
    time_period: str | None = None
    population: str = "전체"
    gold_stat_id: str
    gold_kosis_org_id: str | None = None
    source_case_id: str | None = None


class ComponentVerdictRow(BaseModel):
    row_id: str
    claim_text: str
    stated_value: float
    official_value: float
    unit: str | None = None
    time_period: str | None = None
    expected_verdict: VerdictType
    source_case_id: str | None = None


class OutcomePredictionRecord(BaseModel):
    case_id: str
    expected_verdict: str
    predicted_verdict: str | None = None
    verdict_correct: bool | None = None
    value_within_tolerance: bool | None = None
    predicted_official_value: float | None = None
    reference_stat_id: str | None = None
    predicted_stat_id: str | None = None
    error: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
