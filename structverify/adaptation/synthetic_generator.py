"""
adaptation/synthetic_generator.py — 합성 학습 데이터 자동 생성 (Step 0-2)

[김예슬 - 2026-04-24]
- CANDIDATE_DETECTION_PROMPT 추가:
  · positive: 통계표에서 파생된 검증 가능 수치 주장
  · negative: 의견/일정/감상 등 검증 불가 문장
- _generate_candidate_samples(): candidate detection 학습 데이터 생성
- generate_synthetic_pairs()에 candidate detection 샘플 통합
- _filter_quality(): candidate 샘플 필터 조건 추가

[참고] Self-Instruct (Wang et al., ACL 2023)
[참고] Textbooks Are All You Need (Gunasekar et al., Microsoft, 2023)
"""
from __future__ import annotations

import json
from typing import Any

from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)

# ── 프롬프트 ──────────────────────────────────────────────────────────────

CLAIM_GENERATION_PROMPT = """당신은 한국 뉴스 기자입니다.
아래 공식 통계표 정보를 보고, 이 통계표로 검증할 수 있는 뉴스 주장 {n}개를 생성하세요.

통계표 ID: {stat_id}
통계표명: {stat_name}
발행기관: {org_name}
분류: {category_path}
관련 키워드: {keywords}

규칙:
- 실제 뉴스에 나올법한 자연스러운 한국어 문장
- 반드시 구체적인 수치(%, 만명, 억원, ha 등) 포함
- 주장 유형 다양하게: 증가/감소/규모/비교 중 섞어서
- 검증 가능한 사실적 주장만 (의견/전망 제외)

JSON 배열로만 답하세요:
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

JSON으로만 답하세요:
{{
  "indicator": "측정 지표",
  "time_period": "기준 시점",
  "unit": "단위",
  "population": "대상 범위",
  "value": 수치 또는 null,
  "stat_id": "{stat_id}"
}}"""

CANDIDATE_DETECTION_PROMPT = """아래 통계표 정보를 바탕으로 candidate detection 학습 데이터를 생성하세요.

통계표: {stat_name} ({stat_id})
키워드: {keywords}

다음 두 종류의 문장을 각각 {n}개씩 생성하세요:

[positive] 이 통계표로 검증 가능한 수치 기반 주장:
  - 구체적인 수치/비율/규모 포함
  - 공식 통계와 대조 가능한 사실 주장

[negative] 검증 불가능한 문장 (다양한 유형으로):
  - 의견/감상 (예: "정책이 아쉽다")
  - 단순 이벤트 일정 (예: "박람회가 10월에 열린다")
  - 미래 전망 (예: "줄어들 것으로 보인다")
  - 추상적 주장 (예: "문제가 심각하다")

JSON으로만 답하세요:
{{
  "positives": ["문장1", "문장2", ...],
  "negatives": ["문장1", "문장2", ...]
}}"""


# ── 메인 함수 ─────────────────────────────────────────────────────────────

async def generate_synthetic_pairs(
    catalog: list[dict[str, Any]],
    llm: LLMClient,
    claims_per_table: int = 3,
    max_tables: int | None = None,
) -> list[dict[str, Any]]:
    """
    KOSIS 메타데이터 → LLM Self-Instruct → 학습 쌍 자동 생성.

    생성 태스크:
      1) claim_to_stat   : 주장 → 관련 통계표 매핑
      2) claim_to_schema : 주장 → 구조화 스키마 추출
      3) stat_to_claim   : 통계표 → 주장 판별 (역방향)
      4) candidate_pos   : 검증 후보 문장 (positive)
      5) candidate_neg   : 검증 비후보 문장 (negative)
    """
    tables = catalog[:max_tables] if max_tables else catalog
    logger.info(f"합성 데이터 생성: {len(tables)}개 통계표 × {claims_per_table}쌍")

    all_pairs: list[dict[str, Any]] = []
    success, fail = 0, 0

    for idx, table in enumerate(tables):
        try:
            # claim/schema 쌍 생성
            claims = await _generate_claims(llm, table, claims_per_table)
            for claim_data in claims:
                schema = await _generate_schema(llm, claim_data, table)
                all_pairs.append({
                    "task":         "claim_to_stat",
                    "claim":        claim_data.get("claim", ""),
                    "stat_id":      table["stat_id"],
                    "stat_name":    table["stat_name"],
                    "indicator":    claim_data.get("indicator", ""),
                    "claim_type":   claim_data.get("claim_type", ""),
                    "schema":       schema,
                    "source_table": table,
                })

            # candidate detection 쌍 생성
            candidate_samples = await _generate_candidate_samples(
                llm, table, n=claims_per_table
            )
            all_pairs.extend(candidate_samples)

            success += 1
            if (idx + 1) % 50 == 0:
                logger.info(f"진행: {idx + 1}/{len(tables)} ({len(all_pairs)}쌍)")

        except Exception as e:
            fail += 1
            logger.warning(f"통계표 {table.get('stat_id')} 실패: {e}")

    filtered = _filter_quality(all_pairs)
    logger.info(
        f"합성 데이터 완료: 성공 {success}개, 실패 {fail}개 | "
        f"생성 {len(all_pairs)}쌍 → 필터 후 {len(filtered)}쌍"
    )
    return filtered


# ── 내부 생성 함수 ────────────────────────────────────────────────────────

async def _generate_claims(
    llm: LLMClient, table: dict, n: int
) -> list[dict[str, Any]]:
    """통계표 → 뉴스 주장 N개 생성"""
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
    if isinstance(result, dict) and "raw" not in result:
        return [result]
    return []


async def _generate_schema(
    llm: LLMClient, claim_data: dict, table: dict
) -> dict[str, Any]:
    """주장 → 검증 스키마 추출"""
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


async def _generate_candidate_samples(
    llm: LLMClient, table: dict, n: int = 3
) -> list[dict[str, Any]]:
    """
    통계표 기반 candidate detection 학습 데이터 생성.
    positive/negative 쌍으로 candidate_scorer 학습에 사용.
    """
    prompt = CANDIDATE_DETECTION_PROMPT.format(
        stat_name=table.get("stat_name", ""),
        stat_id=table.get("stat_id", ""),
        keywords=", ".join(table.get("keywords", [])),
        n=n,
    )
    try:
        result = await llm.generate_json(
            prompt=prompt,
            system_prompt="학습 데이터 생성기. JSON으로만 답하세요.",
        )
    except Exception as e:
        logger.warning(f"candidate 샘플 생성 실패 ({table.get('stat_id')}): {e}")
        return []

    samples = []
    stat_id   = table.get("stat_id", "")
    stat_name = table.get("stat_name", "")

    for sent in result.get("positives", []):
        if isinstance(sent, str) and sent.strip():
            samples.append({
                "task":            "candidate_detection",
                "sentence":        sent.strip(),
                "candidate_label": True,
                "stat_id":         stat_id,
                "stat_name":       stat_name,
            })

    for sent in result.get("negatives", []):
        if isinstance(sent, str) and sent.strip():
            samples.append({
                "task":            "candidate_detection",
                "sentence":        sent.strip(),
                "candidate_label": False,
                "stat_id":         stat_id,
                "stat_name":       stat_name,
            })

    return samples


# ── 품질 필터링 ───────────────────────────────────────────────────────────

import re
_NUMERIC_RE = re.compile(r"\d")


def _filter_quality(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    합성 데이터 품질 필터링.

    claim_to_stat 필터:
      - 주장 10자 미만 제거
      - 수치 미포함 제거
      - stat_id 누락 제거
      - 중복 제거

    candidate_detection 필터:
      - 문장 5자 미만 제거
      - positive는 수치 필수
    """
    seen: set[str] = set()
    filtered: list[dict[str, Any]] = []

    for pair in pairs:
        task = pair.get("task", "")

        if task == "candidate_detection":
            sentence = pair.get("sentence", "").strip()
            if len(sentence) < 5:
                continue
            if pair.get("candidate_label") is True and not _NUMERIC_RE.search(sentence):
                continue
            key = f"cand:{sentence}"
            if key in seen:
                continue
            seen.add(key)
            filtered.append(pair)

        else:
            claim = pair.get("claim", "").strip()
            if len(claim) < 10:
                continue
            if not _NUMERIC_RE.search(claim):
                continue
            if not pair.get("stat_id"):
                continue
            key = f"claim:{claim}"
            if key in seen:
                continue
            seen.add(key)
            filtered.append(pair)

    return filtered


async def save_synthetic_data(
    pairs: list[dict[str, Any]],
    output_path: str = "ml/data/synthetic_pretrain.jsonl",
) -> None:
    """합성 데이터 JSONL 저장"""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    task_counts: dict[str, int] = {}
    for p in pairs:
        t = p.get("task", "unknown")
        task_counts[t] = task_counts.get(t, 0) + 1

    logger.info(f"합성 데이터 저장: {output_path} ({len(pairs)}건)")
    logger.info(f"태스크별 분포: {task_counts}")