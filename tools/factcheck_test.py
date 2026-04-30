"""
# 수정자: 박재윤
# 수정 날짜: 2026-04-30
# 수정 내용: RAG 기반 팩트체크 테스트 스크립트 전체 개선

# [DONE] HCX 임베딩 기반 pgvector 유사도 검색
# [DONE] KOSIS API 실제 수치 조회
# [DONE] HCX LLM 팩트체크 판정
# [DONE] KOSIS 통합검색 API로 테이블 검색 (pgvector 폴백)
# [DONE] itmId=ALL, objL1=ALL 기본 설정 + err:20 시 objL 점진 추가
# [DONE] prdSe 순회 (Y→M→Q→newEstPrdCnt) + 시점 포맷 자동 변환
# [DONE] 후보 테이블 복수 시도 (top-1 실패 시 다음 후보)
# [DONE] ThreadPoolExecutor 병렬 처리 (max_workers=3)
# [DONE] CSV 결과 저장 + 요약 출력
# [DONE] 수치 직접 비교 로직 추가 (numeric_check, 단위 자동 변환)
# [DONE] 단위 타입 검증 (is_same_unit_type, 명↔개월 혼용 방지)
# [DONE] 90% 초과 오차 → 판단불가 처리 (테이블 매칭 오류 방지)
# [DONE] matched_table, matched_table_id CSV 컬럼 추가
# [DONE] KOSIS 통합검색 vwCd=MT_ZTITLE 필터 (국가통계만, 지역통계 제외)
# [DONE] HCX 키워드 추출 연도 제거 regex 개선 (4자리 숫자만 제거)
# [DONE] is_relevant_table LLM 제거 → pgvector 유사도 점수로 대체
# [TODO] asyncpg 마이그레이션 (현재 psycopg2)
# [TODO] 빈 벡터(임베딩 실패) 재시도 로직

tools/factcheck_test.py — RAG 기반 팩트체크 테스트 스크립트 (v5)

흐름:
1. claims 테이블에서 claim 조회
2. HCX LLM으로 KOSIS 검색 키워드 + 카테고리 필터 생성
3. KOSIS 통합검색 API(국가통계만) + pgvector 필터 검색 + pgvector 전체 검색
4. 후보 테이블별로:
   a. prdSe=Y → M → Q 순으로 시도 (시점 포맷 자동 변환)
   b. 각 시도마다 err:20이면 objL2~8 점진 추가
   c. 전부 실패 시 newEstPrdCnt=3 (최신 데이터) 폴백
   d. 데이터 조회 성공 시 numeric_check → LLM 재판정 (필요시만)
5. 모든 후보 실패 시 "판단불가" 반환
6. ThreadPoolExecutor로 claim 단위 병렬 처리 (max_workers=3)
7. CSV 저장 + 요약 출력
"""
import os
import re
import json
import csv
import threading
import httpx
import psycopg2
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

HCX_API_KEY = os.getenv("CLOVASTUDIO_API_KEY")
KOSIS_API_KEY = os.getenv("KOSIS_API_KEY")

PG_CONN = dict(
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT"),
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
)

print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


# ────────────────────────────────────────────────────────────
# 1. 카테고리 + 키워드 동시 추출
# ────────────────────────────────────────────────────────────

def extract_category_and_keyword(claim: dict) -> tuple[list[str], str]:
    prompt = f"""다음 뉴스 수치 주장을 분석하여 두 가지를 추출하세요.

원문: {claim['context']}
지표: {claim['field_name']}

1) 이 통계가 속할 KOSIS 카테고리 경로 키워드 2~3개 (쉼표 구분)
   KOSIS 주요 분야: 인구, 가구, 고용, 노동, 임금, 물가, 가계, 보건, 사회, 복지, 교육, 환경, 농림, 수산, 건설, 주택, 토지, 교통, 정보통신, 경제, 산업, 무역
   이 수치가 나올 법한 통계조사명도 포함하세요.

2) KOSIS 통계표 이름에 들어갈 법한 검색어 2~3단어 (숫자/연도 금지)

형식:
카테고리: 키워드1, 키워드2, 키워드3
검색어: 검색 키워드"""

    try:
        resp = httpx.post(
            "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-DASH-002",
            headers={"Authorization": f"Bearer {HCX_API_KEY}", "Content-Type": "application/json"},
            json={"messages": [{"role": "user", "content": prompt}], "maxTokens": 80, "temperature": 0},
            timeout=30,
        )
        content = resp.json()["result"]["message"]["content"].strip()
    except Exception as e:
        return ([], claim['field_name'])

    category_keywords = []
    search_keyword = claim['field_name']

    for line in content.split("\n"):
        line = line.strip()
        if "카테고리" in line and ":" in line:
            cats = line.split(":", 1)[1].strip()
            category_keywords = [c.strip().strip('"\'') for c in cats.split(",") if c.strip()]
        elif "검색어" in line and ":" in line:
            kw = line.split(":", 1)[1].strip().strip('"\'')
            kw = re.sub(r'\b\d{4}\b', '', kw).strip()
            kw = re.sub(r'\s+', ' ', kw)
            if kw:
                search_keyword = kw

    return (category_keywords, search_keyword)


# ────────────────────────────────────────────────────────────
# 2. 테이블 탐색
# ────────────────────────────────────────────────────────────

def search_kosis_api(keyword: str, max_results: int = 5) -> list[dict]:
    try:
        resp = httpx.get(
            "https://kosis.kr/openapi/statisticsSearch.do",
            params={
                "method": "getList", "apiKey": KOSIS_API_KEY,
                "searchNm": keyword, "format": "json", "jsonVD": "Y",
                "resultCount": max_results, "sort": "RANK",
                "vwCd": "MT_ZTITLE",  # ← 주제별 국가통계만
            },
            timeout=30,
        )
        data = resp.json()
    except Exception:
        return []

    if isinstance(data, dict) and ("err" in data or "errMsg" in data):
        return []
    if not isinstance(data, list):
        data = [data] if isinstance(data, dict) and "TBL_ID" in data else []

    return [
        {"stat_id": item["TBL_ID"], "org_id": item["ORG_ID"],
         "stat_name": item.get("TBL_NM", ""), "source": "kosis_search"}
        for item in data if item.get("TBL_ID") and item.get("ORG_ID")
    ]


def get_embedding(text: str) -> list[float]:
    resp = httpx.post(
        "https://clovastudio.stream.ntruss.com/v1/api-tools/embedding/v2",
        headers={"Authorization": f"Bearer {HCX_API_KEY}", "Content-Type": "application/json"},
        json={"text": text},
        timeout=30,
    )
    return resp.json()["result"]["embedding"]


def search_pgvector(embedding: list[float], category_keywords: list[str] = None,
                    field_name_terms: list[str] = None, top_k: int = 5) -> list[dict]:
    conn = psycopg2.connect(**PG_CONN)
    cur = conn.cursor()
    vector_str = "[" + ",".join(map(str, embedding)) + "]"

    where_clauses = []
    where_params = []

    if category_keywords:
        cat_parts = " OR ".join(["category_path ILIKE %s"] * len(category_keywords))
        where_clauses.append(f"({cat_parts})")
        where_params.extend([f"%{kw}%" for kw in category_keywords])

    if field_name_terms:
        name_parts = " OR ".join(["stat_name ILIKE %s"] * len(field_name_terms))
        where_clauses.append(f"({name_parts})")
        where_params.extend([f"%{t}%" for t in field_name_terms])

    if where_clauses:
        where_sql = " OR ".join(where_clauses)
        cur.execute(f"""
            SELECT stat_id, stat_name, category_path, org_id,
                   1 - (embedding <-> %s::vector) AS similarity
            FROM kosis_stat_catalog
            WHERE {where_sql}
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        """, (vector_str, *where_params, vector_str, top_k))
    else:
        cur.execute("""
            SELECT stat_id, stat_name, category_path, org_id,
                   1 - (embedding <-> %s::vector) AS similarity
            FROM kosis_stat_catalog
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        """, (vector_str, vector_str, top_k))

    results = [
        {"stat_id": row[0], "stat_name": row[1], "category_path": row[2],
         "org_id": row[3], "source": "pgvector", "similarity": row[4]}
        for row in cur.fetchall()
    ]
    cur.close()
    conn.close()
    return results


def find_candidate_tables(claim: dict, top_k: int = 5) -> list[dict]:
    category_keywords, search_keyword = extract_category_and_keyword(claim)
    safe_print(f"  🔑 검색 키워드: {search_keyword}")
    safe_print(f"  📂 카테고리 필터: {category_keywords}")

    stopwords = {"수", "명", "원", "건", "개", "만", "천", "억", "이상", "이하", "약", "총", "전체"}
    field_terms = [
        w for w in re.split(r'[\s·,/]+', claim.get('field_name', ''))
        if len(w) >= 2 and w not in stopwords and not re.match(r'^[\d.]+$', w)
    ]

    candidates = search_kosis_api(search_keyword, max_results=top_k)

    combined_keyword = f"{claim['field_name']} {search_keyword}"
    embedding = get_embedding(combined_keyword)

    if category_keywords or field_terms:
        pg_filtered = search_pgvector(embedding, category_keywords, field_terms, top_k=top_k)
    else:
        pg_filtered = []

    pg_unfiltered = search_pgvector(embedding, None, None, top_k=3)

    existing_ids = {c["stat_id"] for c in candidates}
    for r in pg_filtered + pg_unfiltered:
        if r["stat_id"] not in existing_ids:
            candidates.append(r)
            existing_ids.add(r["stat_id"])

    return candidates


# ────────────────────────────────────────────────────────────
# 3. KOSIS 통계자료 조회
# ────────────────────────────────────────────────────────────

def fetch_kosis_data(org_id: str, tbl_id: str, time_reference: str):
    year_match = re.search(r'(\d{4})', time_reference)
    year = year_match.group(1) if year_match else "2024"

    period_strategies = [
        {"prdSe": "Y",  "startPrdDe": year,         "endPrdDe": year,         "label": "연간"},
        {"prdSe": "M",  "startPrdDe": f"{year}01",  "endPrdDe": f"{year}12",  "label": "월간"},
        {"prdSe": "Q",  "startPrdDe": f"{year}01",  "endPrdDe": f"{year}04",  "label": "분기"},
    ]

    for strategy in period_strategies:
        base_params = {
            "method": "getList", "apiKey": KOSIS_API_KEY,
            "format": "json", "jsonVD": "Y",
            "orgId": org_id, "tblId": tbl_id,
            "itmId": "ALL", "objL1": "ALL",
            "prdSe": strategy["prdSe"],
            "startPrdDe": strategy["startPrdDe"],
            "endPrdDe": strategy["endPrdDe"],
        }
        data = _try_with_objL_escalation(base_params)
        if data is not None:
            return data

    for prd in ["Y", "M", "Q"]:
        fallback_params = {
            "method": "getList", "apiKey": KOSIS_API_KEY,
            "format": "json", "jsonVD": "Y",
            "orgId": org_id, "tblId": tbl_id,
            "itmId": "ALL", "objL1": "ALL",
            "prdSe": prd, "newEstPrdCnt": 3,
        }
        data = _try_with_objL_escalation(fallback_params)
        if data is not None:
            return data

    return None


def _try_with_objL_escalation(base_params: dict):
    data = _call_kosis_data(base_params)
    if data is None:
        return None

    if isinstance(data, dict) and data.get("err") == "20":
        for level in range(2, 9):
            key = f"objL{level}"
            if key not in base_params:
                base_params[key] = "ALL"
                data = _call_kosis_data(base_params)
                if data is None:
                    return None
                if not (isinstance(data, dict) and data.get("err") == "20"):
                    break

    if isinstance(data, dict) and "err" in data:
        return None

    return data


def _call_kosis_data(params: dict):
    try:
        resp = httpx.get(
            "https://kosis.kr/openapi/Param/statisticsParameterData.do",
            params=params, timeout=30,
        )
        return resp.json()
    except Exception:
        return None


# ────────────────────────────────────────────────────────────
# 4. LLM 판정
# ────────────────────────────────────────────────────────────

def judge_with_hcx(claim: dict, kosis_data, stat_name: str) -> dict:
    sample = kosis_data[:10] if isinstance(kosis_data, list) else kosis_data

    prompt = f"""다음 뉴스 기사의 수치 주장과 공식 통계 데이터를 비교하여 팩트체크하세요.

[뉴스 주장]
- 지표: {claim['field_name']}
- 수치: {claim['field_value']} {claim['unit']}
- 시점: {claim['time_reference']}
- 원문: {claim['context']}

[공식 통계] KOSIS - {stat_name}
{json.dumps(sample, ensure_ascii=False, indent=2)}

판정 기준:
- "일치": 통계 데이터에서 해당 수치를 직접 확인할 수 있고, 뉴스 주장과 근사하게 맞음
- "불일치": 통계 데이터에 동일 지표의 수치가 존재하고, 뉴스 주장과 수치가 다를 때만 사용
- "판단불가": 통계 데이터가 완전히 다른 지표이거나, 해당 세분화(학력별, 연령별 등) 데이터가 없거나, 수치를 직접 확인할 수 없는 경우

중요: 통계 데이터가 뉴스 주장과 완전히 다른 지표라면 반드시 "판단불가"로 판정하세요.
단위 주의: KOSIS 데이터가 "천명" 단위면 뉴스의 "명" 단위와 1000배 차이가 납니다.

JSON 형식으로만 답하세요:
{{"verdict": "일치|불일치|판단불가", "reason": "근거 설명"}}"""

    try:
        resp = httpx.post(
            "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-DASH-002",
            headers={"Authorization": f"Bearer {HCX_API_KEY}", "Content-Type": "application/json"},
            json={"messages": [{"role": "user", "content": prompt}], "maxTokens": 300, "temperature": 0},
            timeout=30,
        )
        content = resp.json()["result"]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        return {"verdict": "판단불가", "reason": str(e)}

def extract_numeric_values(kosis_data: list) -> list[dict]:
    values = []
    for row in kosis_data:
        dt = row.get("DT", "")
        unit = row.get("UNIT_NM", "")
        prd = row.get("PRD_DE", "")
        try:
            val = float(str(dt).replace(",", ""))
            values.append({"value": val, "unit": unit, "period": prd, "raw": row})
        except:
            continue
    return values


def normalize_value(value: float, kosis_unit: str) -> float:
    kosis_unit = kosis_unit.lower()
    if "천" in kosis_unit:
        return value * 1000
    elif "백만" in kosis_unit or "million" in kosis_unit:
        return value * 1000000
    elif "억" in kosis_unit:
        return value * 100000000
    return value

def is_same_unit_type(claim_unit: str, kosis_unit: str) -> bool:
    claim_unit = claim_unit.lower().strip()
    kosis_unit = kosis_unit.lower().strip()

    # 둘 다 비어있으면 통과
    if not claim_unit or not kosis_unit:
        return True

    people_keywords = ["명", "person", "인구", "가구", "세대"]
    time_keywords = ["개월", "월", "month", "년", "일", "주"]
    ratio_keywords = ["%", "퍼센트", "percent", "율", "비율"]
    money_keywords = ["원", "won", "달러", "dollar", "usd"]

    def get_type(unit):
        for kw in people_keywords:
            if kw in unit: return "people"
        for kw in time_keywords:
            if kw in unit: return "time"
        for kw in ratio_keywords:
            if kw in unit: return "ratio"
        for kw in money_keywords:
            if kw in unit: return "money"
        return "unknown"

    claim_type = get_type(claim_unit)
    kosis_type = get_type(kosis_unit)

    # 둘 다 unknown이면 통과 (판단 불가)
    if claim_type == "unknown" or kosis_type == "unknown":
        return True

    return claim_type == kosis_type


def numeric_check(claim: dict, kosis_data: list) -> dict:
    if not isinstance(kosis_data, list) or not kosis_data:
        return {"verdict": "판단불가", "reason": "KOSIS 데이터 없음"}

    claim_value = float(claim["field_value"])
    claim_unit = claim.get("unit", "")
    year_match = re.search(r'(\d{4})', claim.get("time_reference", ""))
    claim_year = year_match.group(1) if year_match else None

    kosis_values = extract_numeric_values(kosis_data)
    if not kosis_values:
        return {"verdict": "판단불가", "reason": "수치 추출 실패"}

    best_match = None
    best_error = float("inf")

    for kv in kosis_values:
        if claim_year and kv["period"]:
            kv_year = kv["period"][:4]
            try:
                if abs(int(claim_year) - int(kv_year)) > 2:
                    continue
            except:
                pass

        normalized = normalize_value(kv["value"], kv["unit"])
        if normalized == 0:
            continue

        # 단위 타입이 다르면 스킵
        if not is_same_unit_type(claim_unit, kv["unit"]):
            continue

        error_rate = abs(normalized - claim_value) / max(abs(claim_value), 1)
        if error_rate < best_error:
            best_error = error_rate
            best_match = {**kv, "normalized": normalized, "error_rate": error_rate}

    if best_match is None:
        return {"verdict": "판단불가", "reason": "해당 연도 데이터 없음"}

    unit = claim.get("unit", "")
    if best_error <= 0.1:
        return {"verdict": "일치", "reason": f"KOSIS {best_match['normalized']:,.0f}{unit} vs 뉴스 {claim_value:,.0f}{unit} (오차 {best_error*100:.1f}%)"}
    elif best_error <= 0.3:
        return {"verdict": "판단불가", "reason": f"유사하나 오차 {best_error*100:.1f}% — LLM 재판정"}
    elif best_error > 0.9:
        # [v5] - 박재윤: 90% 초과 오차는 테이블 매칭 오류로 판단불가
        return {"verdict": "판단불가", "reason": f"오차 {best_error*100:.1f}% — 테이블 매칭 오류 의심"}
    else:
        return {"verdict": "불일치", "reason": f"KOSIS {best_match['normalized']:,.0f}{unit} vs 뉴스 {claim_value:,.0f}{unit} (오차 {best_error*100:.1f}%)"}


def is_relevant_table(claim: dict, stat_name: str) -> bool:
    """테이블이 claim과 완전히 무관한지 확인 (무관하면 False)"""
    prompt = f"""다음 뉴스 수치 주장과 KOSIS 통계표 이름이 같은 주제를 다루는지 판단하세요.

뉴스 주장 지표: {claim['field_name']}
KOSIS 통계표명: {stat_name}

같은 주제(인구, 고용, 임금, 물가 등 분야가 일치)이면 "YES"
완전히 다른 주제이면 "NO"

YES 또는 NO만 답하세요."""

    try:
        resp = httpx.post(
            "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-DASH-002",
            headers={"Authorization": f"Bearer {HCX_API_KEY}", "Content-Type": "application/json"},
            json={"messages": [{"role": "user", "content": prompt}], "maxTokens": 10, "temperature": 0},
            timeout=30,
        )
        content = resp.json()["result"]["message"]["content"].strip().upper()
        return "NO" not in content  # NO가 있으면 스킵, 나머지는 통과
    except:
        return True
    
# ────────────────────────────────────────────────────────────
# 5. 단일 claim 처리
# ────────────────────────────────────────────────────────────

def process_single_claim(claim: dict) -> dict:
    safe_print(f"\n{'='*60}")
    safe_print(f"📌 Claim: {claim['field_name']} {claim['field_value']} {claim['unit']} ({claim['time_reference']})")
    safe_print(f"📰 원문: {claim['context'][:100]}...")

    candidates = find_candidate_tables(claim, top_k=5)
    if not candidates:
        safe_print("❌ 후보 테이블 없음")
        return {
            "claim_id": claim["claim_id"],
            "field_name": claim["field_name"],
            "field_value": claim["field_value"],
            "unit": claim["unit"],
            "time_reference": claim["time_reference"],
            "verdict": "판단불가",
            "reason": "후보 테이블 없음",
            "matched_table": None,
            "matched_table_id": None,
        }

    best_result = None
    best_table_name = None
    best_table_id = None

    for i, table in enumerate(candidates):
        safe_print(f"\n  [{i+1}/{len(candidates)}] 🔍 {table['stat_name']} ({table['stat_id']}) via {table['source']}")

        # [v5] - 박재윤: 유사도 필터 제거 (L2거리 음수 문제)
        # if table.get("source") == "pgvector" and table.get("similarity", 1) < 0.5:
        #     safe_print(f"  ⏭️  유사도 낮음 ({table.get('similarity', 0):.2f}) 스킵")
        #     continue

        kosis_data = fetch_kosis_data(table["org_id"], table["stat_id"], claim["time_reference"])

        if kosis_data is None:
            safe_print(f"  ⏭️  데이터 없음, 다음 후보 시도...")
            continue

        data_count = len(kosis_data) if isinstance(kosis_data, list) else 1
        safe_print(f"  ✅ KOSIS 데이터 {data_count}건 조회 성공")

        # [v3] - 박재윤: LLM만으로 판정
        # result = judge_with_hcx(claim, kosis_data, table["stat_name"])

        # [v4] - 박재윤: 수치 직접 비교, LLM은 재판정 케이스만
        # elif result["verdict"] == "판단불가":
        #     llm_result = judge_with_hcx(...)

        # [v5] - 박재윤: LLM 판단불가 재판정 제거 (도메인 제약 방지)
        result = numeric_check(claim, kosis_data)
        if result["verdict"] == "판단불가" and "LLM 재판정" in result["reason"]:
            result = judge_with_hcx(claim, kosis_data, table["stat_name"])

        safe_print(f"  ⚖️  판정: {result['verdict']}")
        safe_print(f"  📝 근거: {result['reason']}")

        if result["verdict"] in ("일치", "불일치"):
            best_result = result
            best_table_name = table["stat_name"]
            best_table_id = table["stat_id"]
            break

        if best_result is None:
            best_result = result
            best_table_name = table["stat_name"]
            best_table_id = table["stat_id"]
        safe_print(f"  ↳ 판단불가 → 다음 후보 탐색...")

    verdict = best_result["verdict"] if best_result else "판단불가"
    reason = best_result["reason"] if best_result else f"모든 후보 테이블({len(candidates)}건)에서 데이터를 찾지 못했습니다."

    safe_print(f"\n  ⚖️  최종 판정: {verdict}")

    return {
        "claim_id": claim["claim_id"],
        "field_name": claim["field_name"],
        "field_value": claim["field_value"],
        "unit": claim["unit"],
        "time_reference": claim["time_reference"],
        "verdict": verdict,
        "reason": reason,
        "matched_table": best_table_name,
        "matched_table_id": best_table_id,
    }


# ────────────────────────────────────────────────────────────
# 6. 메인 실행
# ────────────────────────────────────────────────────────────

def run_factcheck(limit: int = None):
    conn = psycopg2.connect(**PG_CONN)
    cur = conn.cursor()
    if limit:
        cur.execute("""
            SELECT claim_id, field_name, field_value, unit, time_reference, context
            FROM claims LIMIT %s
        """, (limit,))
    else:
        cur.execute("""
            SELECT claim_id, field_name, field_value, unit, time_reference, context
            FROM claims
        """)
    claims = [
        {"claim_id": r[0], "field_name": r[1], "field_value": r[2],
         "unit": r[3], "time_reference": r[4], "context": r[5]}
        for r in cur.fetchall()
    ]
    cur.close()
    conn.close()

    safe_print(f"총 {len(claims)}건 처리 시작 (병렬 3개)")

    results_log = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_single_claim, claim): claim for claim in claims}
        for future in as_completed(futures):
            try:
                result = future.result()
                results_log.append(result)
            except Exception as e:
                claim = futures[future]
                safe_print(f"⚠️ 처리 실패: {claim['field_name']} — {e}")
                results_log.append({
                    "claim_id": claim["claim_id"],
                    "field_name": claim["field_name"],
                    "field_value": claim["field_value"],
                    "unit": claim["unit"],
                    "time_reference": claim["time_reference"],
                    "verdict": "판단불가",
                    "reason": f"처리 중 오류: {e}",
                })

    # CSV 저장
    filename = f"factcheck_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results_log[0].keys())
        writer.writeheader()
        writer.writerows(results_log)

    total = len(results_log)
    match = sum(1 for r in results_log if r["verdict"] == "일치")
    mismatch = sum(1 for r in results_log if r["verdict"] == "불일치")
    unknown = sum(1 for r in results_log if r["verdict"] == "판단불가")

    print(f"\n{'='*60}")
    print(f"📊 최종 결과 요약")
    print(f"  전체: {total}건")
    print(f"  ✅ 일치: {match}건 ({match/total*100:.1f}%)")
    print(f"  ❌ 불일치: {mismatch}건 ({mismatch/total*100:.1f}%)")
    print(f"  ❓ 판단불가: {unknown}건 ({unknown/total*100:.1f}%)")
    print(f"  💾 저장: {filename}")


if __name__ == "__main__":
    run_factcheck()