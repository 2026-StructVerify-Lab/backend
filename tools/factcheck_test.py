"""
# 수정자: 박재윤
# 수정 날짜: 2026-04-29
# 수정 내용: RAG 기반 팩트체크 테스트 스크립트 — 개선판

# [FIX] KOSIS 통합검색 API로 테이블 검색 (pgvector 폴백)
# [FIX] itmId=ALL, objL1=ALL 기본 설정 + err:20 시 objL 점진 추가
# [FIX] prdSe 순회 (Y→M→Q→newEstPrdCnt) + 시점 포맷 자동 변환
# [FIX] 후보 테이블 복수 시도 (top-1 실패 시 다음 후보)
# [FIX] HCX 키워드: 통계조사명 유도, 숫자/연도 자동 제거

tools/factcheck_test.py — RAG 기반 팩트체크 테스트 스크립트 (v2)

흐름:
1. claims 테이블에서 claim 조회
2. HCX LLM으로 KOSIS 검색 키워드 생성 (숫자/연도 자동 제거)
3. KOSIS 통합검색 API로 후보 테이블 탐색 (실패 시 pgvector 폴백)
4. 후보 테이블별로:
   a. prdSe=Y → M → Q 순으로 시도 (시점 포맷 자동 변환)
   b. 각 시도마다 err:20이면 objL2~8 점진 추가
   c. 전부 실패 시 newEstPrdCnt=3 (최신 데이터) 폴백
   d. 데이터가 나오면 LLM 판정 → 종료
5. 모든 후보 실패 시 "판단불가" 반환
"""
import os
import re
import json
import httpx
import psycopg2
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

# ────────────────────────────────────────────────────────────
# 1. 카테고리 + 키워드 동시 추출
# ────────────────────────────────────────────────────────────

def extract_category_and_keyword(claim: dict) -> tuple[list[str], str]:
    """HCX로 claim → (카테고리 필터 키워드, 검색 키워드) 동시 추출.

    카테고리 키워드: category_path ILIKE 필터에 사용 (26만→수백건 축소)
    검색 키워드: pgvector 임베딩 유사도 + KOSIS 통합검색에 사용
    """
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
        print(f"  ⚠️  추출 실패: {e}")
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
            kw = re.sub(r'[\d,.]+[년만명%천백억조개월일]?', '', kw).strip()
            kw = re.sub(r'\s+', ' ', kw)
            if kw:
                search_keyword = kw

    return (category_keywords, search_keyword)


# ────────────────────────────────────────────────────────────
# 2. 테이블 탐색: 카테고리 필터 + pgvector + KOSIS 통합검색
# ────────────────────────────────────────────────────────────

def search_kosis_api(keyword: str, max_results: int = 5) -> list[dict]:
    """KOSIS 통합검색 API"""
    try:
        resp = httpx.get(
            "https://kosis.kr/openapi/statisticsSearch.do",
            params={
                "method": "getList", "apiKey": KOSIS_API_KEY,
                "searchNm": keyword, "format": "json", "jsonVD": "Y",
                "resultCount": max_results, "sort": "RANK",
            },
            timeout=30,
        )
    except Exception as e:
        print(f"  ⚠️  KOSIS 통합검색 요청 실패: {e}")
        return []
    try:
        data = resp.json()
    except Exception:
        return []

    if isinstance(data, dict) and ("err" in data or "errMsg" in data):
        print(f"  ⚠️  KOSIS 통합검색 에러: {data}")
        return []
    if not isinstance(data, list):
        data = [data] if isinstance(data, dict) and "TBL_ID" in data else []

    return [
        {"stat_id": item["TBL_ID"], "org_id": item["ORG_ID"],
         "stat_name": item.get("TBL_NM", ""), "source": "kosis_search"}
        for item in data if item.get("TBL_ID") and item.get("ORG_ID")
    ]


def get_embedding(text: str) -> list[float]:
    """HCX 임베딩 v2"""
    resp = httpx.post(
        "https://clovastudio.stream.ntruss.com/v1/api-tools/embedding/v2",
        headers={"Authorization": f"Bearer {HCX_API_KEY}", "Content-Type": "application/json"},
        json={"text": text},
        timeout=30,
    )
    return resp.json()["result"]["embedding"]


def search_pgvector(embedding: list[float], category_keywords: list[str] = None,
                    field_name_terms: list[str] = None, top_k: int = 5) -> list[dict]:
    """pgvector 유사도 검색.
    
    category_keywords: category_path ILIKE 필터
    field_name_terms: stat_name ILIKE 필터 (claim의 핵심 단어로 테이블명 직접 매칭)
    두 필터는 OR로 결합 — 어느 쪽이든 걸리면 후보에 포함.
    """
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
            SELECT stat_id, stat_name, category_path, org_id
            FROM kosis_stat_catalog
            WHERE {where_sql}
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        """, (*where_params, vector_str, top_k))
    else:
        cur.execute("""
            SELECT stat_id, stat_name, category_path, org_id
            FROM kosis_stat_catalog
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        """, (vector_str, top_k))

    results = [
        {"stat_id": row[0], "stat_name": row[1], "category_path": row[2],
         "org_id": row[3], "source": "pgvector"}
        for row in cur.fetchall()
    ]
    cur.close()
    conn.close()
    return results


def find_candidate_tables(claim: dict, top_k: int = 5) -> list[dict]:
    """카테고리 필터 + field_name 필터 + KOSIS 통합검색 + pgvector로 후보 테이블을 찾는다."""
    category_keywords, search_keyword = extract_category_and_keyword(claim)
    print(f"🔑 검색 키워드: {search_keyword}")
    print(f"📂 카테고리 필터: {category_keywords}")

    # field_name에서 의미 있는 단어 추출 (1글자, 숫자, 일반 단위 제외)
    stopwords = {"수", "명", "원", "건", "개", "만", "천", "억", "이상", "이하", "약", "총", "전체"}
    field_terms = [
        w for w in re.split(r'[\s·,/]+', claim.get('field_name', ''))
        if len(w) >= 2 and w not in stopwords and not re.match(r'^[\d.]+$', w)
    ]
    if field_terms:
        print(f"🏷️  필드명 필터: {field_terms}")

    # ① KOSIS 통합검색
    candidates = search_kosis_api(search_keyword, max_results=top_k)
    if candidates:
        print(f"  ✅ KOSIS 통합검색: {len(candidates)}건")

    # ② pgvector — 카테고리 + field_name 필터 적용
    combined_keyword = f"{claim['field_name']} {search_keyword}"
    embedding = get_embedding(combined_keyword)
    if category_keywords or field_terms:
        pg_filtered = search_pgvector(embedding, category_keywords, field_terms, top_k=top_k)
        print(f"  ✅ pgvector (필터 적용): {len(pg_filtered)}건")
    else:
        pg_filtered = []

    # ③ pgvector — 필터 없이 보충 (다른 경로에 있을 수 있으니)
    pg_unfiltered = search_pgvector(embedding, None, None, top_k=3)

    # 중복 제거하며 합치기 (KOSIS → 필터된 pgvector → 필터없는 pgvector)
    existing_ids = {c["stat_id"] for c in candidates}
    for r in pg_filtered + pg_unfiltered:
        if r["stat_id"] not in existing_ids:
            candidates.append(r)
            existing_ids.add(r["stat_id"])

    return candidates


# ────────────────────────────────────────────────────────────
# 3. KOSIS 통계자료 조회 (itmId + objL 자동 탐색)
# ────────────────────────────────────────────────────────────

def fetch_kosis_data(org_id: str, tbl_id: str, time_reference: str) -> list | dict | None:
    """KOSIS 통계자료 API로 실제 수치를 조회한다.
    
    전략:
    1) prdSe=Y (연간) 시도
    2) err:30이면 prdSe=M (월간, 해당 연도 전체) 시도
    3) err:30이면 prdSe=Q (분기) 시도
    4) 전부 실패하면 newEstPrdCnt=3 (최신 3개 시점) 으로 폴백
    
    각 prdSe 시도마다 err:20이면 objL 레벨을 점진 추가.
    """
    # 연도 추출 (예: "2024년" → "2024", "2023년 3월" → "2023")
    year_match = re.search(r'(\d{4})', time_reference)
    year = year_match.group(1) if year_match else "2024"

    # prdSe별 시점 포맷 정의
    period_strategies = [
        {"prdSe": "Y",  "startPrdDe": year,         "endPrdDe": year,         "label": "연간"},
        {"prdSe": "M",  "startPrdDe": f"{year}01",  "endPrdDe": f"{year}12",  "label": "월간"},
        {"prdSe": "Q",  "startPrdDe": f"{year}01",  "endPrdDe": f"{year}04",  "label": "분기"},
    ]

    for strategy in period_strategies:
        base_params = {
            "method": "getList",
            "apiKey": KOSIS_API_KEY,
            "format": "json",
            "jsonVD": "Y",
            "orgId": org_id,
            "tblId": tbl_id,
            "itmId": "ALL",
            "objL1": "ALL",
            "prdSe": strategy["prdSe"],
            "startPrdDe": strategy["startPrdDe"],
            "endPrdDe": strategy["endPrdDe"],
        }

        data = _try_with_objL_escalation(base_params)

        if data is not None:
            return data

        print(f"    ↳ {strategy['label']}({strategy['prdSe']}) 실패")

    # 최후 수단: 주기 지정 없이 최신 데이터 3건
    print(f"  🔄 최신 데이터 폴백 (newEstPrdCnt=3)...")
    for prd in ["Y", "M", "Q"]:
        fallback_params = {
            "method": "getList",
            "apiKey": KOSIS_API_KEY,
            "format": "json",
            "jsonVD": "Y",
            "orgId": org_id,
            "tblId": tbl_id,
            "itmId": "ALL",
            "objL1": "ALL",
            "prdSe": prd,
            "newEstPrdCnt": 3,
        }
        data = _try_with_objL_escalation(fallback_params)
        if data is not None:
            return data

    return None


def _try_with_objL_escalation(base_params: dict):
    """주어진 params로 호출하고, err:20이면 objL을 점진 추가. 성공 시 data, 실패 시 None."""
    data = _call_kosis_data(base_params)
    if data is None:
        return None

    # err:20 → objL 부족
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

    # 성공인지 체크
    if isinstance(data, dict) and "err" in data:
        return None

    return data


def _call_kosis_data(params: dict):
    """KOSIS Param API 호출 + JSON 파싱. 네트워크/파싱 실패 시 None."""
    try:
        resp = httpx.get(
            "https://kosis.kr/openapi/Param/statisticsParameterData.do",
            params=params,
            timeout=30,
        )
    except Exception as e:
        print(f"  ⚠️  KOSIS API 요청 실패: {e}")
        return None
    try:
        return resp.json()
    except Exception:
        print(f"  ⚠️  KOSIS 응답 JSON 파싱 실패 (status={resp.status_code})")
        return None


# ────────────────────────────────────────────────────────────
# 4. LLM 판정
# ────────────────────────────────────────────────────────────

def judge_with_hcx(claim: dict, kosis_data, stat_name: str) -> dict:
    """HCX LLM으로 일치/불일치 판정"""
    # 데이터가 너무 크면 앞부분만 잘라서 전달
    if isinstance(kosis_data, list):
        sample = kosis_data[:10]
    else:
        sample = kosis_data

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
- "불일치": 통계 데이터에서 동일한 지표의 수치를 확인할 수 있지만, 뉴스 주장과 수치가 다름
- "판단불가": 통계 데이터가 완전히 다른 지표이거나, 해당 수치를 직접 확인할 수 없음

중요: 통계 데이터가 뉴스 주장과 완전히 다른 지표(예: 고용률 vs 인구수)라면 반드시 "판단불가"로 판정하세요.
단, 비경제활동인구 데이터 안에 '쉬었음' 활동상태가 포함되어 있다면 관련 데이터로 간주하세요.

JSON 형식으로만 답하세요:
{{"verdict": "일치|불일치|판단불가", "reason": "근거 설명"}}"""

    resp = httpx.post(
        "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-DASH-002",
        headers={"Authorization": f"Bearer {HCX_API_KEY}", "Content-Type": "application/json"},
        json={
            "messages": [{"role": "user", "content": prompt}],
            "maxTokens": 300,
            "temperature": 0,
        },
        timeout=30,
    )
    content = resp.json()["result"]["message"]["content"]
    try:
        return json.loads(content)
    except Exception:
        return {"verdict": "판단불가", "reason": content}


# ────────────────────────────────────────────────────────────
# 5. 메인 실행
# ────────────────────────────────────────────────────────────

def run_factcheck(limit: int = 3):
    """claims DB에서 claim 꺼내서 팩트체크 실행"""
    conn = psycopg2.connect(**PG_CONN)
    cur = conn.cursor()
    cur.execute("""
        SELECT claim_id, field_name, field_value, unit, time_reference, context
        FROM claims
        LIMIT %s
    """, (limit,))
    claims = [
        {
            "claim_id": row[0],
            "field_name": row[1],
            "field_value": row[2],
            "unit": row[3],
            "time_reference": row[4],
            "context": row[5],
        }
        for row in cur.fetchall()
    ]
    cur.close()
    conn.close()

    for claim in claims:
        print(f"\n{'='*60}")
        print(f"📌 Claim: {claim['field_name']} {claim['field_value']} {claim['unit']} ({claim['time_reference']})")
        print(f"📰 원문: {claim['context'][:100]}...")

        # 후보 테이블 탐색
        candidates = find_candidate_tables(claim, top_k=5)
        if not candidates:
            print("❌ 후보 테이블 없음")
            continue

        # 후보를 순회하면서 판정
        best_result = None
        for i, table in enumerate(candidates):
            print(f"\n  [{i+1}/{len(candidates)}] 🔍 {table['stat_name']} ({table['stat_id']}) via {table['source']}")

            kosis_data = fetch_kosis_data(table["org_id"], table["stat_id"], claim["time_reference"])

            if kosis_data is None:
                print(f"  ⏭️  데이터 없음, 다음 후보 시도...")
                continue

            data_count = len(kosis_data) if isinstance(kosis_data, list) else 1
            print(f"  ✅ KOSIS 데이터 {data_count}건 조회 성공")

            # LLM 판정
            result = judge_with_hcx(claim, kosis_data, table["stat_name"])
            print(f"  ⚖️  판정: {result['verdict']}")
            print(f"  📝 근거: {result['reason']}")

            # "일치" 또는 "불일치"면 확정 종료
            if result["verdict"] in ("일치", "불일치"):
                best_result = result
                break

            # "판단불가"면 저장하고 다음 후보 계속 시도
            if best_result is None:
                best_result = result
            print(f"  ↳ 판단불가 → 다음 후보에서 더 나은 데이터 탐색...")

        if best_result is None:
            print(f"\n  ⚖️  최종 판정: 판단불가")
            print(f"  📝 근거: 모든 후보 테이블({len(candidates)}건)에서 데이터를 찾지 못했습니다.")
        elif best_result["verdict"] == "판단불가":
            print(f"\n  ⚖️  최종 판정: 판단불가")
            print(f"  📝 근거: {best_result['reason']}")


if __name__ == "__main__":
    run_factcheck(limit=3)