"""
adaptation/synthetic_generator.py — 합성 학습 데이터 자동 생성 (Step 0-2)

KOSIS 메타데이터를 LLM에 넣어 학습 쌍을 자동 생성한다. 사람 라벨링 0건.
통계표 1만개 × 3쌍 = 학습 데이터 3만 건.

[참고] Self-Instruct (Wang et al., ACL 2023)
  - https://github.com/yizhongw/self-instruct
  - 논문: Wang, Y., et al. (2023). Self-Instruct: Aligning Language Models
    with Self-Generated Instructions. ACL 2023.
  - seed 데이터로부터 LLM이 instruction-following 데이터를 자동 생성
  - 본 프로젝트에서는 KOSIS 메타데이터를 seed로, 뉴스 주장↔통계표 매핑 쌍을 생성

[참고] Textbooks Are All You Need (Gunasekar et al., Microsoft, 2023)
  - arXiv 2306.11644
  - GPT-3.5로 합성 교과서 데이터를 생성하여 phi-1 모델 학습
  - 고품질 합성 데이터가 대규모 웹 데이터보다 효과적일 수 있음을 증명
  - 본 프로젝트에서 품질 필터링 전략에 참고

사용법:
  python -m adaptation.synthetic_generator     # CLI 실행
  await generate_synthetic_pairs(catalog, llm) # 코드에서 호출
"""
from __future__ import annotations

import json
from typing import Any

from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


# ── 합성 데이터 생성 프롬프트 ──

CLAIM_GENERATION_PROMPT = """당신은 한국 뉴스 기자입니다.
아래 공식 통계표 정보를 보고, 이 통계표로 검증할 수 있는 뉴스 주장 {n}개를 생성하세요.

통계표 ID: {stat_id}
통계표명: {stat_name}
발행기관: {org_name}
분류: {category_path}
관련 키워드: {keywords}

규칙:
- 실제 뉴스에 나올법한 자연스러운 한국어 문장으로 작성
- 반드시 구체적인 수치(%, 만명, 억원, ha 등)를 포함
- 주장 유형을 다양하게: 증가/감소/규모/비교 중 섞어서
- 검증 가능한 사실적 주장만 (의견/전망 제외)

JSON 배열로 답하세요:
[
  {{
    "claim": "뉴스에 나올법한 주장 문장 (수치 포함)",
    "indicator": "검증할 핵심 지표명",
    "stat_id": "{stat_id}",
    "claim_type": "increase|decrease|scale|comparison",
    "expected_unit": "%, 만명, ha 등"
  }}
]"""

SCHEMA_GENERATION_PROMPT = """아래 뉴스 주장에서 검증에 필요한 핵심 정보를 추출하세요.

주장: "{claim}"
관련 통계표: {stat_name} ({stat_id})

JSON으로 답하세요:
{{
  "indicator": "측정 지표",
  "time_period": "기준 시점",
  "unit": "단위",
  "population": "대상 범위",
  "value": 수치 또는 null,
  "stat_id": "{stat_id}"
}}"""


async def generate_synthetic_pairs(
    catalog: list[dict[str, Any]],
    llm: LLMClient,
    claims_per_table: int = 3,
    max_tables: int | None = None,
) -> list[dict[str, Any]]:
    """
    통계표 메타데이터 → LLM → 학습 쌍 자동 생성 (Self-Instruct)

    Args:
        catalog: kosis_crawler가 수집한 통계표 메타데이터 리스트
        llm: LLM 클라이언트 (HCX-003 권장)
        claims_per_table: 통계표당 생성할 주장 수 (기본 3)
        max_tables: 처리할 최대 통계표 수 (None이면 전체)

    Returns:
        합성 학습 데이터 리스트
        [
          {"task": "claim_generation",
           "claim": "농가 고령화율이 60%를 넘었다",
           "stat_id": "DT_1EA1019",
           "indicator": "고령화비율",
           "claim_type": "scale"},
          ...
        ]
    """
    tables = catalog[:max_tables] if max_tables else catalog
    logger.info(f"합성 데이터 생성 시작: {len(tables)}개 통계표 × {claims_per_table}쌍")

    all_pairs: list[dict[str, Any]] = []
    success, fail = 0, 0

    for idx, table in enumerate(tables):
        try:
            # Step 1: 주장 생성
            claims = await _generate_claims(llm, table, claims_per_table)

            # Step 2: 각 주장에 대한 스키마 생성
            for claim_data in claims:
                schema = await _generate_schema(llm, claim_data, table)
                pair = {
                    "task": "claim_to_stat",
                    "claim": claim_data.get("claim", ""),
                    "stat_id": table["stat_id"],
                    "stat_name": table["stat_name"],
                    "indicator": claim_data.get("indicator", ""),
                    "claim_type": claim_data.get("claim_type", ""),
                    "schema": schema,
                    "source_table": table,
                }
                all_pairs.append(pair)

            success += 1
            if (idx + 1) % 50 == 0:
                logger.info(f"진행: {idx + 1}/{len(tables)} ({len(all_pairs)}쌍 생성)")

        except Exception as e:
            fail += 1
            logger.warning(f"통계표 {table.get('stat_id')} 실패: {e}")

    # Step 3: 품질 필터링
    filtered = _filter_quality(all_pairs)

    logger.info(
        f"합성 데이터 생성 완료: "
        f"성공 {success}개 테이블, 실패 {fail}개, "
        f"생성 {len(all_pairs)}쌍 → 필터 후 {len(filtered)}쌍"
    )
    return filtered


async def _generate_claims(
    llm: LLMClient,
    table: dict[str, Any],
    n: int,
) -> list[dict[str, Any]]:
    """
    단일 통계표에 대해 뉴스 주장 N개를 LLM으로 생성한다.

    [참고] Self-Instruct의 instruction generation 단계
    seed(통계표 메타) → LLM → 새로운 instruction(뉴스 주장) 생성
    """
    prompt = CLAIM_GENERATION_PROMPT.format(
        n=n,
        stat_id=table.get("stat_id", ""),
        stat_name=table.get("stat_name", ""),
        org_name=table.get("org_name", ""),
        category_path=table.get("category_path", ""),
        keywords=", ".join(table.get("keywords", [])),
    )

    result = await llm.generate_json(
        prompt=prompt,
        system_prompt="한국 뉴스 기자. JSON 배열로만 답하세요.",
    )

    # 응답이 리스트인지 확인
    if isinstance(result, list):
        return result
    elif isinstance(result, dict) and "raw" in result:
        # JSON 파싱 실패 시 빈 리스트
        return []
    return [result]


async def _generate_schema(
    llm: LLMClient,
    claim_data: dict[str, Any],
    table: dict[str, Any],
) -> dict[str, Any]:
    """
    생성된 주장에 대해 검증 스키마를 LLM으로 추출한다.

    이 스키마가 Schema Induction 학습 데이터가 된다:
    (주장 텍스트 → 구조화된 스키마 JSON)
    """
    prompt = SCHEMA_GENERATION_PROMPT.format(
        claim=claim_data.get("claim", ""),
        stat_name=table.get("stat_name", ""),
        stat_id=table.get("stat_id", ""),
    )

    try:
        return await llm.generate_json(
            prompt=prompt,
            system_prompt="통계 분석 전문가. JSON으로만 답하세요.",
        )
    except Exception:
        return {}


def _filter_quality(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    생성된 합성 데이터의 품질을 필터링한다.

    [참고] Textbooks Are All You Need (Gunasekar et al., 2023)
    합성 데이터의 품질 필터링이 모델 성능에 큰 영향을 미침

    필터 기준:
    1. 주장 텍스트가 비어있거나 너무 짧은 경우 제거
    2. 수치가 포함되지 않은 주장 제거
    3. stat_id가 없는 경우 제거
    4. 중복 주장 제거 (완전 일치 기준)
    """
    import re
    numeric_pattern = re.compile(r"\d")

    seen_claims: set[str] = set()
    filtered: list[dict[str, Any]] = []

    for pair in pairs:
        claim = pair.get("claim", "").strip()

        # 빈 문장 / 너무 짧은 문장
        if len(claim) < 10:
            continue

        # 수치 미포함
        if not numeric_pattern.search(claim):
            continue

        # stat_id 누락
        if not pair.get("stat_id"):
            continue

        # 중복 제거
        if claim in seen_claims:
            continue
        seen_claims.add(claim)

        filtered.append(pair)

    return filtered


async def save_synthetic_data(
    pairs: list[dict[str, Any]],
    output_path: str = "ml/data/synthetic_pretrain.jsonl",
) -> None:
    """
    합성 학습 데이터를 JSONL 파일로 저장한다.

    TODO: S3/MinIO에도 백업 저장
    TODO: DVC로 버전 관리
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(f"합성 데이터 저장: {output_path} ({len(pairs)}건)")


# ── CLI 진입점 ──
if __name__ == "__main__":
    import asyncio

    async def main():
        from structverify.adaptation.kosis_crawler import crawl_kosis_catalog

        logger.info("=== 합성 학습 데이터 생성 시작 ===")
        llm = LLMClient({"provider": "hcx", "model": "HCX-003"})

        # 1) KOSIS 메타 수집 (또는 이미 수집된 데이터 로드)
        catalog = await crawl_kosis_catalog()

        # 2) 합성 데이터 생성
        pairs = await generate_synthetic_pairs(
            catalog, llm, claims_per_table=3, max_tables=100  # 테스트: 100개만
        )

        # 3) 저장
        await save_synthetic_data(pairs)
        logger.info(f"=== 완료: {len(pairs)}건 생성 ===")

    asyncio.run(main())
