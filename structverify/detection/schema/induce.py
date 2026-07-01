"""detection/schema/induce.py — 스키마 유도 LLM 호출 및 후처리.

schema_inductor.py에서 분리 (로직 move-only, 동작 변경 없음).

[v6.14] source_phrase 검증 — context leak 방지
[v6.14 E] value/prev_value 환산 교정
[v6.15 L] 차이 schema prev_value 역산
[2026-05-21] value_role 자동 추론, aggregation 필드
"""
from __future__ import annotations

import re

from structverify.core.schemas import ClaimSchema
from structverify.detection.prompts.schema import (
    CLAIM_SCHEMA_LIST_JSON_SCHEMA,
    SCHEMA_INDUCTION_PROMPT,
)
from structverify.detection.schema.validate import (
    _extract_numbers_from_text,
    _safe_float,
    _source_phrase_in_claim,
    _validate_schema,
    _value_in_claim_text,
    _verify_and_correct_value,
)
from structverify.utils.llm_client import LLMClient
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


async def _induce_multiple(
    llm: LLMClient,
    claim_text: str,
    domain: str = "general",
    domain_hint: str = "",
    context: str = "",
    temporal_hint: str = "",
    anchor_year: int | None = None,
) -> list[ClaimSchema]:
    """
    단일 주장 → list[ClaimSchema] (0개 이상).

    LLM이 한 문장의 모든 검증 가능 수치를 schemas 배열로 반환.
    [v6.14] source_phrase 검증으로 context leak 방지:
      - LLM이 schema마다 source_phrase 제공 (예: '2만 171명', '6.7%')
      - source_phrase가 claim_text에 *substring으로 등장*하는지 검증
      - 등장하지 않으면 (context leak) 그 schema 폐기
    Structured Outputs 사용 — JSON 파싱 실패 없음.
    """
    prompt = SCHEMA_INDUCTION_PROMPT.format(
        claim_text=claim_text,
        context=context or claim_text,
        domain=domain,
        domain_hint=domain_hint,
        temporal_hint=temporal_hint,
    )

    try:
        r = await llm.generate_structured(
            prompt=prompt,
            schema=CLAIM_SCHEMA_LIST_JSON_SCHEMA,
            system_prompt=(
                "통계 분석 전문가. 위 규칙을 엄격히 따르세요. "
                "★ 핵심: [검증 대상 문장]에 literally 등장하는 수치만 추출. "
                "[문맥]의 수치는 절대 추출 금지. "
                "각 schema에 source_phrase 의무 포함. "
                "★ '예상/예보/전망/예측' 포함 indicator는 KOSIS 검증 불가 → 해당 schema 추출 금지."
            ),
        )
    except Exception as e:
        logger.warning(f"스키마 유도 LLM 호출 예외: {e}")
        return []

    # [2026-05-25] LLM thought 디버깅용 — LLM이 어떤 schema 후보를 *왜* 추출했는지
    # 화면 UI에서 보기 어려운 케이스 대응. 응답 원본 json을 INFO로 펼쳐 보여줌.
    try:
        import json as _json
        logger.info(
            f"[schema_inductor] LLM 응답 본문 (claim_text={claim_text[:60]!r}...) ↓\n"
            f"────── SCHEMA RESPONSE START ──────\n"
            f"{_json.dumps(r, ensure_ascii=False, indent=2)}\n"
            f"────── SCHEMA RESPONSE END ──────"
        )
    except Exception:
        logger.info(f"[schema_inductor] LLM 응답 raw: {str(r)[:1000]}")

    schemas_raw = r.get("schemas") or []
    if not isinstance(schemas_raw, list):
        logger.warning(f"스키마 유도: schemas가 list 아님 ({type(schemas_raw)})")
        return []

    results: list[ClaimSchema] = []
    for item in schemas_raw:
        if not isinstance(item, dict):
            continue

        # ── [v6.14] source_phrase 검증 (context leak 방지) ──
        source_phrase = (item.get("source_phrase") or "").strip()
        if source_phrase:
            if not _source_phrase_in_claim(source_phrase, claim_text):
                logger.warning(
                    f"  ⚠️ context leak 감지: source_phrase={source_phrase!r} "
                    f"가 검증 대상 문장에 없음 → schema 폐기 "
                    f"(indicator={item.get('indicator')}, value={item.get('value')})"
                )
                continue
        else:
            # source_phrase 없으면 value 자체로 검증 (LLM이 의무 위반한 경우 fallback)
            val = item.get("value")
            if val is not None and not _value_in_claim_text(val, claim_text):
                logger.warning(
                    f"  ⚠️ context leak 의심 (source_phrase 누락): "
                    f"value={val} 가 문장에 없음 → schema 폐기 "
                    f"(indicator={item.get('indicator')})"
                )
                continue

        try:
            # [v6.14 E fix] value 환산 정확성 검증 + 자동 교정
            # LLM이 "2만 171" → 21710 같은 환산 오류를 내는 경우 발견됨.
            # source_phrase가 있으면 거기서 직접 환산값 추출 → LLM value와 비교 → 5% 이상 차이면 교정.
            raw_value = _safe_float(item.get("value"))
            corrected_value, was_corrected = _verify_and_correct_value(
                raw_value, source_phrase
            )

            # [2026-05-21] value=null fallback — LLM이 value를 빠뜨려도 source_phrase에서
            # 숫자 복원. 도메인 무관, 한국어 키워드 하드코딩 X.
            # ("0.79명" → 0.79, "20717명" → 20717, "1만 2741개" → 12741, "23만 8천명" → 238000)
            #
            # [22:40 진단] "1만 2741개"는 _extract_numbers_from_text가 {1, 2741, 12741}처럼
            # 한글 단위 정합값(12741)뿐 아니라 부분 숫자(1, 2741)도 같이 반환해 *set 크기 >1*이
            # 되어 폴백 미적용 → schema value=None → 서울/경기 claim이 모두 unverifiable로 떨어짐.
            # 정답: 한글 단위 결합값(가장 큰 수)이 거의 항상 의도된 value임. set이 여러 개면 max 사용.
            if corrected_value is None and source_phrase:
                _fallback_nums = _extract_numbers_from_text(source_phrase)
                if _fallback_nums:
                    # 단일 숫자거나 한글 단위 결합값(=max)가 그 의미 — 둘 다 max로 통합.
                    _picked = max(_fallback_nums)
                    if len(_fallback_nums) == 1:
                        _msg = f"단일 숫자 {_picked} 복원"
                    else:
                        _msg = (
                            f"숫자 {len(_fallback_nums)}개 중 최대값 {_picked} 복원 "
                            f"(한글 단위 결합 추정, 후보={sorted(_fallback_nums)})"
                        )
                    logger.warning(
                        f"  🔧 value=null 폴백: source_phrase={source_phrase!r}에서 "
                        f"{_msg} (LLM이 value 누락)"
                    )
                    corrected_value = float(_picked)
            if was_corrected:
                logger.warning(
                    f"  🔧 value 환산 교정: LLM={raw_value} → 교정={corrected_value} "
                    f"(source_phrase={source_phrase!r}, indicator={item.get('indicator')})"
                )

            # [v6.14 C2] prev_value 검증 + ClaimSchema 생성
            # schemas.py에 prev_value 필드가 *아직 없는 경우* import 안 깨지게 try/except
            prev_value_raw = item.get("prev_value")
            prev_value = _safe_float(prev_value_raw) if prev_value_raw is not None else None
            prev_time_period = (item.get("prev_time_period") or "").strip() or None
            prev_phrase = (item.get("prev_phrase") or "").strip()

            # prev_phrase가 있으면 검증 (context leak 방지 — source_phrase와 동일 규칙)
            # [v6.17] prev_value/prev_phrase만 폐기하고 prev_time_period는 유지.
            #   시점은 본문 표현("1년 전")에서 계산한 것이라 leak이 아니며,
            #   prev_value가 없어도 시점만 있으면 통계 DB에서 그 시점 값을
            #   직접 조회해 검증할 수 있음.
            if prev_phrase and not _source_phrase_in_claim(prev_phrase, claim_text):
                logger.warning(
                    f"  ⚠️ prev_phrase context leak: {prev_phrase!r} 가 검증 대상 문장에 없음 "
                    f"→ prev_value만 폐기 (prev_time_period={prev_time_period!r}는 유지, "
                    f"indicator={item.get('indicator')})"
                )
                prev_value = None
                prev_phrase = None
                # prev_time_period는 일부러 유지 — 검증 단계에서 사용

            # prev_phrase가 있으면 prev_value 환산 정확성도 검증 (E fix 응용)
            if prev_phrase and prev_value is not None:
                corrected_prev, prev_was_corrected = _verify_and_correct_value(
                    prev_value, prev_phrase
                )
                if prev_was_corrected:
                    logger.warning(
                        f"  🔧 prev_value 환산 교정: LLM={prev_value} → 교정={corrected_prev} "
                        f"(prev_phrase={prev_phrase!r})"
                    )
                prev_value = corrected_prev

            # [v6.16] time_period가 비어있으면 문서 anchor_year로 채움
            #   "전국 공시가격 상승률 4.5%" 처럼 시점 표현이 없는 문장도
            #   기사 작성연도(anchor_year) 기준으로 검증되도록 보정.
            _tp = item.get("time_period") or None
            if not _tp and anchor_year is not None:
                _tp = str(anchor_year)
                logger.info(
                    f"  [시점 보정] time_period 없음 → anchor_year={anchor_year} 적용 "
                    f"(indicator={item.get('indicator')})"
                )

            schema_kwargs = dict(
                indicator=item.get("indicator") or None,
                time_period=_tp,
                unit=item.get("unit") or None,
                population=item.get("population") or None,
                value=corrected_value,
                source_reference=item.get("source_reference") or None,
                graph_schema_candidates=item.get("graph_schema_candidates") or [],
                parent_path=item.get("parent_path") or None,
                is_approximate=bool(item.get("is_approximate", False)),
                modifier=item.get("modifier") or None,
            )

            # [2026-05-21] aggregation 필드 추출 — null-safe, 도메인 무관.
            # LLM이 "평균/총합/최근 N년" 류 신호를 감지해 채우며 한국어 키워드 하드코딩 X.
            # 모두 None이면 일반 base/derived 흐름으로 폴백. ClaimSchema 구버전 호환은 try/except.
            _agg_op_raw = item.get("aggregation")
            _agg_op = str(_agg_op_raw).strip().lower() if _agg_op_raw else None
            if _agg_op in ("", "null", "none"):
                _agg_op = None
            _agg_window_raw = item.get("aggregation_window")
            try:
                _agg_window = int(_agg_window_raw) if _agg_window_raw is not None else None
                if _agg_window is not None and _agg_window <= 0:
                    _agg_window = None
            except (TypeError, ValueError):
                _agg_window = None
            _agg_range_raw = item.get("aggregation_time_range")
            if isinstance(_agg_range_raw, list):
                _agg_range = [str(x).strip() for x in _agg_range_raw if x is not None and str(x).strip()]
                _agg_range = _agg_range or None
            else:
                _agg_range = None
            try:
                ClaimSchema.model_fields["aggregation"]
                schema_kwargs["aggregation"] = _agg_op
                schema_kwargs["aggregation_window"] = _agg_window
                schema_kwargs["aggregation_time_range"] = _agg_range
            except KeyError:
                # 구버전 ClaimSchema — 무시
                if _agg_op:
                    logger.warning(
                        f"  ℹ️ aggregation={_agg_op!r} 추출됐으나 ClaimSchema에 필드 없음. "
                        f"core/schemas.py에 aggregation/aggregation_window/aggregation_time_range 추가 필요."
                    )
            # prev_* 필드는 schemas.py에 *추가됐을 때만* 전달
            # (구버전 schemas.py와 backward compat)
            try:
                ClaimSchema.model_fields["prev_value"]  # 필드 존재 여부 확인
                schema_kwargs["prev_value"] = prev_value
                schema_kwargs["prev_time_period"] = prev_time_period
                schema_kwargs["prev_phrase"] = prev_phrase or None
            except KeyError:
                # schemas.py에 prev_* 필드 없음 — 무시
                if prev_value is not None:
                    logger.warning(
                        f"  ℹ️ prev_value={prev_value} 추출됐으나 ClaimSchema에 필드 없음. "
                        f"core/schemas.py에 prev_value/prev_time_period/prev_phrase 필드 추가 필요."
                    )

            # [2026-05-21] value_role 자동 추론 — schema_inductor가 분기한 *이유*를
            # downstream planner에 명시적으로 전달. LLM이 같은 claim_text를 보고
            # base/derived를 헷갈리는 걸 방지.
            #
            # [K 패치 2026-05-21] indicator suffix 우선 검사. prev_value 유무는
            #   2차 신호로 격하. 합계출산율 0.79처럼 LLM이 *base 절대값 schema*에
            #   prev_value=0.73을 추가 정보로 박아도 *indicator에 ~증가/~차이 같은
            #   derived suffix가 없으면 base*로 분류.
            #
            #   - indicator suffix(~증가율/~비율 류) + 비율 단위 → derived_rate
            #   - indicator suffix(~증가/~감소/~차이 류, 비율 아님)    → derived_difference
            #   - 그 외 → base (prev_value 있어도 base — 단일 값 검증)
            try:
                ClaimSchema.model_fields["value_role"]
                _ind = (item.get("indicator") or "").strip()
                _unit = (item.get("unit") or "").strip()
                _RATE_SUFFIXES = (
                    "증가율", "감소율", "증감률", "변화율", "상승률", "하락률",
                    "비율", "비중",
                )
                _DIFF_SUFFIXES = (
                    "증가", "감소", "증감", "변화", "차이",
                )
                _is_rate_indicator = any(_ind.endswith(s) for s in _RATE_SUFFIXES)
                _is_pct_unit = _unit in ("%", "퍼센트", "퍼센트포인트", "%p")
                _is_diff_indicator = (
                    any(_ind.endswith(s) for s in _DIFF_SUFFIXES)
                    and not _is_rate_indicator
                )
                # [2026-05-21] aggregation 우선 분기 — LLM이 aggregation 연산자를 채웠으면
                # base/derived 분류보다 우선. 도메인 무관 (LLM이 의미 판단).
                _has_agg = bool(_agg_op) or bool(_agg_window) or bool(_agg_range)
                if _has_agg:
                    schema_kwargs["value_role"] = "aggregation"
                    # aggregation은 단일 시점이 아닌 N개 시점 fetch이므로 prev_*는 의미 없음 → clear
                    if schema_kwargs.get("prev_value") is not None or schema_kwargs.get("prev_time_period"):
                        logger.info(
                            f"  [U] aggregation 분류 → prev_value/prev_time_period clear "
                            f"(indicator={_ind!r}, agg={_agg_op!r}, window={_agg_window!r})"
                        )
                        schema_kwargs["prev_value"] = None
                        schema_kwargs["prev_time_period"] = None
                        schema_kwargs["prev_phrase"] = None
                elif _is_rate_indicator or _is_pct_unit:
                    schema_kwargs["value_role"] = "derived_rate"
                elif _is_diff_indicator:
                    schema_kwargs["value_role"] = "derived_difference"
                else:
                    # base — indicator suffix가 derived가 아니면 prev_value 유무
                    # 무관하게 base. prev_value는 후처리에서 *clear* 해서 reflect
                    # LLM이 자율 prev fetch 시도하지 않도록 한다.
                    schema_kwargs["value_role"] = "base"
                    if schema_kwargs.get("prev_value") is not None or schema_kwargs.get("prev_time_period"):
                        logger.info(
                            f"  [K] base 분류 → prev_value/prev_time_period clear "
                            f"(indicator={_ind!r}, was prev_value={schema_kwargs.get('prev_value')!r})"
                        )
                        schema_kwargs["prev_value"] = None
                        schema_kwargs["prev_time_period"] = None
                        schema_kwargs["prev_phrase"] = None
            except KeyError:
                pass  # value_role 필드 없는 구버전 — 무시

            schema = ClaimSchema(**schema_kwargs)
        except Exception as e:
            logger.debug(f"개별 schema 파싱 실패: {e}")
            continue

        if _validate_schema(schema):
            results.append(schema)

    # ── [v6.15 L fix] 차이 schema의 prev_value 자동 역산 ─────────────────
    # 같은 sentence에서 *절대값 schema*와 *차이 schema*가 함께 나왔을 때,
    # 차이 schema의 prev_value가 *비어있으면* → 절대값 − 차이값으로 역산.
    #
    # 예: "합계출산율 0.79명으로 지난해보다 0.06명 증가"
    #   - 절대값 schema: 합계출산율=0.79
    #   - 차이 schema: 합계출산율 차이=0.06, prev_value=None
    #   → 자동 역산: prev_value = 0.79 - 0.06 = 0.73
    #
    # 효과: verifier C2 분기가 작동 → KOSIS 절대값 row와 자동 계산 비교.

    def _prev_year_period(tp: str | None) -> str | None:
        """[수정 v6.23] time_period에서 '1년 전' 시점을 계산.

        '차이/증감' claim의 비교 기준은 보통 '지난해 같은 달/기간'이다.
        prev_time_period를 현재 시점과 똑같이 두면(이전 버그) prev와
        current가 같아져 → fetch가 두 시점을 못 가져오고 검증 불가.
        연도만 1 빼고 월/분기 부분은 그대로 둔다. 도메인 무관.

        '2025-04' → '2024-04'  /  '2023' → '2022'  /  '2025Q2' → '2024Q2'
        """
        if not tp:
            return None
        s = str(tp).strip()
        import re as _re
        m = _re.match(r"^(\d{4})(.*)$", s)
        if not m:
            return None
        try:
            year = int(m.group(1))
        except ValueError:
            return None
        return f"{year - 1}{m.group(2)}"

    try:
        _has_prev_field = "prev_value" in ClaimSchema.model_fields
    except Exception:
        _has_prev_field = False

    if _has_prev_field and len(results) >= 2:
        # 절대값 schema (indicator에 "차이/증감/변화량" 없음) 찾기
        abs_schemas = [
            s for s in results
            if s.indicator and not any(
                kw in s.indicator for kw in ("차이", "증감", "변화량", "증가율")
            ) and s.value is not None
        ]
        diff_schemas = [
            s for s in results
            if s.indicator and ("차이" in s.indicator or "증감" in s.indicator
                                or "변화량" in s.indicator)
            and s.value is not None
            and getattr(s, "prev_value", None) is None
        ]

        for diff_s in diff_schemas:
            # 같은 indicator base 찾기 (예: "합계출산율 차이" → "합계출산율")
            diff_base = diff_s.indicator
            for kw in ("차이", "증감", "변화량"):
                diff_base = diff_base.replace(kw, "").strip()

            # 매칭되는 절대값 schema
            matching_abs = None
            for abs_s in abs_schemas:
                if abs_s.indicator and (abs_s.indicator == diff_base
                                        or diff_base in abs_s.indicator
                                        or abs_s.indicator in diff_base):
                    # 단위도 비슷한지 (둘 다 비어있거나 둘 다 있고 같은 type)
                    if (not abs_s.unit and not diff_s.unit) or \
                       (abs_s.unit and diff_s.unit and abs_s.unit == diff_s.unit):
                        matching_abs = abs_s
                        break

            if matching_abs:
                # 역산: prev = current - diff
                derived_prev = matching_abs.value - diff_s.value
                # [수정 v6.23] prev_time_period — '지난해 같은 달/기간'이므로
                # 현재 시점에서 1년 전으로 계산. (이전 버그: 현재 시점을
                # 그대로 넣어 prev==current → fetch가 두 시점 확보 실패 →
                # difference claim이 '단일 fetch로 검증 불가'로 끝남)
                # LLM이 prev_time_period를 채웠으면 그대로 존중, 비었으면 역산.
                _llm_prev_tp = getattr(diff_s, "prev_time_period", None)
                _derived_prev_tp = _llm_prev_tp or _prev_year_period(
                    matching_abs.time_period
                )
                # diff_s에 prev_value 채워넣기 (model_copy)
                try:
                    updated = diff_s.model_copy(update={
                        "prev_value": derived_prev,
                        "prev_time_period": _derived_prev_tp,
                        "prev_phrase": None,  # 역산이라 원문 phrase 없음
                    })
                    # results 안에서 교체
                    for i, s in enumerate(results):
                        if s is diff_s:
                            results[i] = updated
                            break
                    logger.info(
                        f"  ✨ prev_value 역산 (L fix): {diff_s.indicator}={diff_s.value} "
                        f"← {matching_abs.indicator}={matching_abs.value} - {diff_s.value} "
                        f"= {derived_prev:.4f} (prev_time={_derived_prev_tp})"
                    )
                except Exception as e:
                    logger.debug(f"prev_value 역산 실패: {e}")

    return results
