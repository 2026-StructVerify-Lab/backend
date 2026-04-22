"""
adaptation/synthetic_generator.py — 합성 학습 데이터 자동 생성

변경 요약
- positive claim 생성뿐 아니라
  candidate detection용 negative / weak negative 샘플도 함께 생성
- sentence_to_candidate 태스크 학습이 가능하도록 출력 구조 확장
"""
from __future__ import annotations

import json
import re
from typing import Any

from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

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
- 검증 가능한 사실 주장만 작성

JSON 배열로 답하세요:
[
  {{
    "claim": "뉴스에 나올법한 주장 문장",
    "indicator": "핵심 지표명",
    "claim_type": "increase|decrease|scale|comparison",
    "expected_unit": "%, 만명, ha 등"
  }}
]
"""

NEGATIVE_CANDIDATE_PROMPT = """당신은 한국 뉴스 기자입니다.
아래 통계표와 관련된 기사 문맥에서, "숫자가 있거나 기사 문장처럼 보이지만 공식 통계 검증 후보로는 부적절한 문장" {n}개를 생성하세요.

통계표 ID: {stat_id}
통계표명: {stat_name}
발행기관: {org_name}
분류: {category_path}
관련 키워드: {keywords}

규칙:
- 뉴스 기사 문장처럼 자연스러운 한국어
- 단순 일정 소개, 행사 설명, 발언 소개, 맥락 설명, 애매한 표현 등을 사용
- 공식 통계와 바로 매핑하기 어렵게 작성
- 일부 문장은 숫자를 포함해도 됨

JSON 배열로 답하세요:
[
  {{
    "claim": "검증 후보로는 부적절한 문장"
  }}
]
"""

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
}}
"""


async def generate_synthetic_pairs(
    catalog: list[dict[str, Any]],
    llm: LLMClient,
    claims_per_table: int = 3,
    max_tables: int | None = None,
) -> list[dict[str, Any]]:
    tables = catalog[:max_tables] if max_tables else catalog
    logger.info(f"합성 데이터 생성 시작: {len(tables)}개 통계표")

    all_pairs: list[dict[str, Any]] = []

    for idx, table in enumerate(tables):
        try:
            positive_claims = await _generate_claims(llm, table, claims_per_table)
            negative_claims = await _generate_negative_candidates(llm, table, max(1, claims_per_table // 2))

            # positive candidate
            for claim_data in positive_claims:
                claim_text = claim_data.get("claim", "").strip()
                if not claim_text:
                    continue

                schema = await _generate_schema(llm, claim_data, table)
                all_pairs.append({
                    "task": "sentence_to_candidate",
                    "claim": claim_text,
                    "stat_id": table.get("stat_id", ""),
                    "stat_name": table.get("stat_name", ""),
                    "indicator": claim_data.get("indicator", ""),
                    "claim_type": claim_data.get("claim_type", ""),
                    "schema": schema,
                    "candidate_label": True,
                    "candidate_score": 0.95,
                    "candidate_signals": {
                        "source": "synthetic_positive",
                        "verifiable_by_stat": True,
                    },
                    "source_table": table,
                })

            # negative candidate
            for claim_data in negative_claims:
                claim_text = claim_data.get("claim", "").strip()
                if not claim_text:
                    continue

                all_pairs.append({
                    "task": "sentence_to_candidate",
                    "claim": claim_text,
                    "stat_id": "",
                    "stat_name": "",
                    "indicator": "",
                    "claim_type": "",
                    "schema": {},
                    "candidate_label": False,
                    "candidate_score": 0.05,
                    "candidate_signals": {
                        "source": "synthetic_negative",
                        "verifiable_by_stat": False,
                    },
                    "source_table": table,
                })

            if (idx + 1) % 50 == 0:
                logger.info(f"진행: {idx + 1}/{len(tables)}")

        except Exception as e:
            logger.warning(f"통계표 {table.get('stat_id')} 처리 실패: {e}")

    return _filter_quality(all_pairs)


async def _generate_claims(
    llm: LLMClient,
    table: dict[str, Any],
    n: int,
) -> list[dict[str, Any]]:
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
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        return [result]
    return []


async def _generate_negative_candidates(
    llm: LLMClient,
    table: dict[str, Any],
    n: int,
) -> list[dict[str, Any]]:
    prompt = NEGATIVE_CANDIDATE_PROMPT.format(
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
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        return [result]
    return []


async def _generate_schema(
    llm: LLMClient,
    claim_data: dict[str, Any],
    table: dict[str, Any],
) -> dict[str, Any]:
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
    품질 필터링

    positive는 수치가 없어도 schema가 있으면 통과 가능
    negative는 너무 짧은 문장만 제거
    """
    filtered: list[dict[str, Any]] = []
    seen_claims: set[str] = set()
    numeric_pattern = re.compile(r"\d")

    for pair in pairs:
        claim = pair.get("claim", "").strip()
        if len(claim) < 8:
            continue

        if claim in seen_claims:
            continue
        seen_claims.add(claim)

        is_positive = bool(pair.get("candidate_label", False))
        if is_positive:
            has_schema = bool(pair.get("schema"))
            has_numeric = bool(numeric_pattern.search(claim))
            if not has_schema and not has_numeric:
                continue

        filtered.append(pair)

    return filtered


async def save_synthetic_data(
    pairs: list[dict[str, Any]],
    output_path: str = "ml/data/synthetic_pretrain.jsonl",
) -> None:
    """
    기존 저장 함수 그대로 써도 되지만, JSONL 저장은 유지한다.
    """
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"합성 데이터 저장 완료: {output_path} ({len(pairs)}건)")