"""
tests/test_step3_5_detection.py — Step 3~5 Detection 파이프라인 테스트

실행 방법:
    # API 키 없이 되는 단위 테스트
    python -m pytest tests/test_step3_5_detection.py -k "unit" -v

    # NCP 키 필요 (실제 LLM 호출)
    export NCP_API_KEY="nv-xxx"
    python -m pytest tests/test_step3_5_detection.py -k "llm" -v -s
"""
import os
import pytest
from uuid import uuid4

HAS_NCP_KEY = bool(os.environ.get("NCP_API_KEY"))
skip_no_ncp = pytest.mark.skipif(not HAS_NCP_KEY, reason="NCP_API_KEY 없음")


# ── 공통 fixture ───────────────────────────────────────────────────────────

@pytest.fixture
def llm_config():
    return {
        "provider": "openai",           # hcx → openai 로 변경
        "models": {
            "heavy": "gpt-4o",
            "light": "gpt-4o-mini",     # HCX-DASH 대신 gpt-4o-mini
        },
        "temperature": 0.1,
        "max_tokens": 1024,
        "openai_key_env": "OPENAI_API_KEY",  # openai 키 env 이름
    }

@pytest.fixture
def sample_sir_doc():
    """농업 도메인 SIR 문서 fixture"""
    from structverify.core.schemas import (
        SIRDocument, SIRBlock, Sentence, BlockType, SourceType
    )
    return SIRDocument(
        source_type=SourceType.TEXT,
        blocks=[
            SIRBlock(
                block_id="b0000",
                type=BlockType.HEADING,
                content="농가 고령화 심화…65세 이상 비율 64% 넘어",
                sentences=[
                    Sentence(
                        sent_id="s0000",
                        text="농가 고령화 심화…65세 이상 비율 64% 넘어",
                        has_numeric_surface=True,
                        candidate_score=0.85,
                        candidate_label=True,
                    )
                ],
            ),
            SIRBlock(
                block_id="b0001",
                type=BlockType.PARAGRAPH,
                content="통계청에 따르면 2023년 기준 65세 이상 과수 농가 경영주 비율이 64.2%에 달해 역대 최고치를 기록했다.",
                sentences=[
                    Sentence(
                        sent_id="s0001",
                        text="통계청에 따르면 2023년 기준 65세 이상 과수 농가 경영주 비율이 64.2%에 달해 역대 최고치를 기록했다.",
                        has_numeric_surface=True,
                        candidate_score=0.92,
                        candidate_label=True,
                    ),
                    Sentence(
                        sent_id="s0002",
                        text="농촌 인구 감소가 지속되고 있어 대책 마련이 시급하다는 지적이 나온다.",
                        has_numeric_surface=False,
                        candidate_score=0.2,
                        candidate_label=False,
                    ),
                ],
            ),
        ],
    )


@pytest.fixture
def sample_claim():
    """검증용 Claim fixture"""
    from structverify.core.schemas import Claim, SourceOffset
    return Claim(
        doc_id=uuid4(),
        block_id="b0001",
        sent_id="s0001",
        claim_text="2023년 기준 65세 이상 과수 농가 경영주 비율이 64.2%에 달해 역대 최고치를 기록했다.",
        check_worthy_score=0.92,
    )


# ════════════════════════════════════════════════════════
# Step 3: domain_classifier 단위 테스트
# ════════════════════════════════════════════════════════

def test_unit_supported_domains():
    """DomainRegistry: 시드 도메인 로드 및 format_for_prompt 확인"""
    from structverify.detection.domain_classifier import DomainRegistry, DEFAULT_SEED_DOMAINS
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = DomainRegistry(f"{tmpdir}/registry.yaml")

        # 파일 없을 때 → 시드 도메인으로 초기화
        domains = registry.load()
        assert "agriculture" in domains
        assert "general" in domains

        # format_for_prompt 형식 확인
        prompt_str = registry.format_for_prompt()
        assert "- agriculture:" in prompt_str
        assert "농림수산식품" in prompt_str


def test_unit_registry_register_new():
    """DomainRegistry.register(): 새 도메인 저장 + 중복 무시"""
    from structverify.detection.domain_classifier import DomainRegistry
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = DomainRegistry(f"{tmpdir}/registry.yaml")
        registry.register("real_estate", "부동산 (아파트, 매매가, 전세)")

        domains = registry.load()
        assert "real_estate" in domains
        assert domains["real_estate"] == "부동산 (아파트, 매매가, 전세)"

        # 중복 등록 → 무시 (설명 변경 안 됨)
        registry.register("real_estate", "다른 설명")
        assert registry.load()["real_estate"] == "부동산 (아파트, 매매가, 전세)"


def test_unit_build_text_preview_heading_first(sample_sir_doc):
    """_build_text_preview: heading 블록 우선 포함"""
    from structverify.detection.domain_classifier import _build_text_preview
    preview = _build_text_preview(sample_sir_doc)
    assert "고령화" in preview
    assert len(preview) <= 600


def test_unit_load_domain_pack_missing():
    """_load_domain_pack: 파일 없어도 None 반환 (에러 X)"""
    from structverify.detection.domain_classifier import _load_domain_pack
    result = _load_domain_pack("nonexistent_domain", {})
    assert result is None


@skip_no_ncp
async def test_llm_classify_agriculture(sample_sir_doc, llm_config):
    """LLM 도메인 분류 — 농업 문서 → agriculture"""
    from structverify.detection.domain_classifier import classify_domain
    import tempfile

    config = {
        "llm": llm_config,
        "domain_registry_path": f"{tempfile.mkdtemp()}/registry.yaml",
    }
    domain, description = await classify_domain(sample_sir_doc, config)
    print(f"\n[도메인 분류 결과] domain={domain}, description={description}")
    assert isinstance(domain, str)
    assert isinstance(description, str)
    assert len(description) > 0
    assert sample_sir_doc.detected_domain == domain


@skip_no_ncp
async def test_llm_classify_economy(llm_config):
    """LLM 도메인 분류 — 경제 문서"""
    from structverify.core.schemas import SIRDocument, SIRBlock, BlockType, SourceType
    from structverify.detection.domain_classifier import classify_domain
    import tempfile

    doc = SIRDocument(
        source_type=SourceType.TEXT,
        blocks=[SIRBlock(
            block_id="b0",
            type=BlockType.PARAGRAPH,
            content="한국은행은 올해 경제성장률 전망을 2.3%로 하향 조정했다. GDP 성장이 둔화될 것으로 보인다.",
        )],
    )
    config = {
        "llm": llm_config,
        "domain_registry_path": f"{tempfile.mkdtemp()}/registry.yaml",
    }
    domain, description = await classify_domain(doc, config)
    print(f"\n[경제 도메인 분류] domain={domain}, description={description}")
    assert domain in ("economy", "finance", "general")


@skip_no_ncp
async def test_llm_classify_new_domain(llm_config):
    """LLM 도메인 분류 — 신규 도메인 생성 + 레지스트리 저장"""
    from structverify.core.schemas import SIRDocument, SIRBlock, BlockType, SourceType
    from structverify.detection.domain_classifier import classify_domain, DomainRegistry
    import tempfile

    tmpdir = tempfile.mkdtemp()
    registry_path = f"{tmpdir}/registry.yaml"

    doc = SIRDocument(
        source_type=SourceType.TEXT,
        blocks=[SIRBlock(
            block_id="b0",
            type=BlockType.PARAGRAPH,
            content="수도권 아파트 평균 매매가가 8억을 돌파하며 역대 최고치를 기록했다. 전세가도 동반 상승 중이다.",
        )],
    )
    config = {"llm": llm_config, "domain_registry_path": registry_path}
    domain, description = await classify_domain(doc, config)
    print(f"\n[신규 도메인] domain={domain}, description={description}")

    # 레지스트리에 저장됐는지 확인
    registry = DomainRegistry(registry_path)
    registered = registry.load()
    print(f"[레지스트리] {list(registered.keys())}")
    assert domain in registered


# ════════════════════════════════════════════════════════
# Step 4: candidate_scorer 단위 테스트
# ════════════════════════════════════════════════════════

def test_unit_heuristic_high_score():
    """heuristic fallback — 수치+시점+대상+비교 모두 포함 시 높은 점수"""
    from structverify.detection.candidate_scorer import _score_candidate_heuristic
    score, label, source, signals = _score_candidate_heuristic(
        "2023년 과수 농가 경영주 64.2% 증가",
        threshold=0.65,
    )
    assert score >= 0.65
    assert label is True
    assert source == "heuristic_fallback"
    assert signals["has_quantity"] is True
    assert signals["has_time_expr"] is True


def test_unit_heuristic_low_score():
    """heuristic fallback — 의견 문장 낮은 점수"""
    from structverify.detection.candidate_scorer import _score_candidate_heuristic
    score, label, source, _ = _score_candidate_heuristic(
        "농촌 고령화 문제는 심각하게 고민해야 합니다.",
        threshold=0.65,
    )
    assert label is False


@skip_no_ncp
async def test_llm_candidate_score_positive(llm_config):
    """LLM candidate scoring — 검증 가능 주장 높은 점수"""
    from structverify.detection.candidate_scorer import score_candidate
    score, label, source, signals = await score_candidate(
        sentence="2023년 기준 65세 이상 과수 농가 경영주 비율이 64.2%에 달했다.",
        config={"llm": llm_config, "candidate_detection": {"threshold": 0.65}},
    )
    print(f"\n[positive candidate] score={score:.2f}, label={label}, source={source}")
    print(f"  signals: {signals}")
    assert score >= 0.5  # 수치+시점+대상 있으므로 높아야 함


@skip_no_ncp
async def test_llm_candidate_score_negative(llm_config):
    """LLM candidate scoring — 의견 문장 낮은 점수"""
    from structverify.detection.candidate_scorer import score_candidate
    score, label, source, _ = await score_candidate(
        sentence="농촌 고령화 문제에 대한 정부의 대책 마련이 시급하다는 목소리가 높다.",
        config={"llm": llm_config, "candidate_detection": {"threshold": 0.65}},
    )
    print(f"\n[negative candidate] score={score:.2f}, label={label}")
    assert score < 0.8  # 의견 문장이므로 낮아야 함


# ════════════════════════════════════════════════════════
# Step 4: claim_detector 단위/통합 테스트
# ════════════════════════════════════════════════════════

@skip_no_ncp
async def test_llm_detect_claims_full(sample_sir_doc, llm_config):
    """claim_detector end-to-end — SIR 문서에서 주장 탐지"""
    from structverify.detection.claim_detector import detect_claims

    config = {
        "llm": llm_config,
        "candidate_detection": {"threshold": 0.5, "teacher_llm_fallback": True},
        "verification": {"min_confidence": 0.5},
    }

    claims = await detect_claims(sample_sir_doc, config)
    print(f"\n[탐지된 주장 수] {len(claims)}")
    for c in claims:
        print(f"  [{c.sent_id}] score={c.check_worthy_score:.2f}: {c.claim_text[:60]}")

    # 농업 고령화 수치 문장은 탐지되어야 함
    assert isinstance(claims, list)


# ════════════════════════════════════════════════════════
# Step 5: schema_inductor 단위 테스트
# ════════════════════════════════════════════════════════

def test_unit_safe_float_numeric():
    """_safe_float: 다양한 수치 표현 파싱"""
    from structverify.detection.schema_inductor import _safe_float
    assert _safe_float(64.2) == 64.2
    assert _safe_float("64.2") == 64.2
    assert _safe_float("64.2%") == 64.2
    assert _safe_float("약 64") == 64.0
    assert _safe_float(None) is None
    assert _safe_float("없음") is None


def test_unit_validate_schema_ok():
    """_validate_schema: 유효한 스키마"""
    from structverify.detection.schema_inductor import _validate_schema
    from structverify.core.schemas import ClaimSchema
    schema = ClaimSchema(indicator="고령화비율", value=64.2, unit="%")
    assert _validate_schema(schema) is True


def test_unit_validate_schema_no_indicator():
    """_validate_schema: indicator 없으면 실패"""
    from structverify.detection.schema_inductor import _validate_schema
    from structverify.core.schemas import ClaimSchema
    schema = ClaimSchema(indicator=None, value=64.2)
    assert _validate_schema(schema) is False


def test_unit_validate_schema_short_indicator():
    """_validate_schema: indicator 1자면 실패"""
    from structverify.detection.schema_inductor import _validate_schema
    from structverify.core.schemas import ClaimSchema
    schema = ClaimSchema(indicator="율")
    assert _validate_schema(schema) is False


@skip_no_ncp
async def test_llm_induce_schema_agriculture(sample_claim, llm_config):
    """LLM 스키마 유도 — 농업 고령화 주장"""
    from structverify.detection.schema_inductor import induce_schemas

    config = {"llm": llm_config, "detected_domain": "agriculture"}
    result = await induce_schemas([sample_claim], config)

    claim = result[0]
    print(f"\n[스키마 유도 결과]")
    print(f"  indicator  : {claim.schema.indicator if claim.schema else 'None'}")
    print(f"  time_period: {claim.schema.time_period if claim.schema else 'None'}")
    print(f"  unit       : {claim.schema.unit if claim.schema else 'None'}")
    print(f"  population : {claim.schema.population if claim.schema else 'None'}")
    print(f"  value      : {claim.schema.value if claim.schema else 'None'}")
    print(f"  graph_candidates: {len(claim.schema.graph_schema_candidates) if claim.schema else 0}개")

    assert claim.schema is not None
    assert claim.schema.indicator is not None
    assert claim.schema.value == pytest.approx(64.2, abs=1.0)


# ════════════════════════════════════════════════════════
# Step 3~5 통합 테스트 (파이프라인 전체)
# ════════════════════════════════════════════════════════

@skip_no_ncp
async def test_llm_pipeline_step3_to_5(sample_sir_doc, llm_config):
    """Step 3~5 전체 파이프라인 통합 테스트"""
    from structverify.detection.domain_classifier import classify_domain
    from structverify.detection.claim_detector import detect_claims
    from structverify.detection.schema_inductor import induce_schemas

    config = {
        "llm": llm_config,
        "candidate_detection": {"threshold": 0.5, "teacher_llm_fallback": True},
        "verification": {"min_confidence": 0.5},
    }

    # Step 3
    domain, description = await classify_domain(sample_sir_doc, config)
    config["detected_domain"] = domain
    print(f"\n[Step 3] 도메인: {domain} — {description}")
    assert isinstance(domain, str)

    # Step 4
    claims = await detect_claims(sample_sir_doc, config)
    print(f"[Step 4] 탐지된 주장: {len(claims)}건")

    # Step 5
    if claims:
        claims = await induce_schemas(claims, config)
        claim = claims[0]
        print(f"[Step 5] 스키마: indicator={claim.schema.indicator if claim.schema else None}")

    print("\n[통합 테스트 완료]")