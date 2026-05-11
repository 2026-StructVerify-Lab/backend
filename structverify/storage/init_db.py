#  아래 스크립 트 파일 CreateTable, DbConnect,openapiyes/noSave.test.py 전부 베이스라인 코드에 침투예정.

"""
# 수정자: 박재윤
# 수정 날짜: 2026-04-26
# 수정 내용: requests, claims, truths, results 테이블 생성 스크립트 구현

# [DONE] CREATE TABLE 4개 구현
# [DONE] pgvector extension 추가
# [DONE] kosis_stat_catalog 테이블 추가
"""
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT"),
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD")
)
print("PostgreSQL 연결 확인")
print(f"Host: {os.getenv('POSTGRES_HOST')}, Port: {os.getenv('POSTGRES_PORT')}, DB: {os.getenv('POSTGRES_DB')}, User: {os.getenv('POSTGRES_USER')}")


cur = conn.cursor()

cur.execute("SELECT inet_server_addr(), inet_server_port();")
print("SERVER:", cur.fetchone())

cur.execute("SELECT extname FROM pg_extension;")
print("EXTENSIONS:", cur.fetchall())

cur.execute("""

    CREATE TABLE IF NOT EXISTS requests (
        request_id VARCHAR PRIMARY KEY,
        source_type VARCHAR,
        domain VARCHAR,
        raw_data JSONB,
        submitted_at TIMESTAMP DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS claims (
        claim_id VARCHAR PRIMARY KEY,
        request_id VARCHAR REFERENCES requests(request_id),
        field_name VARCHAR,
        field_value FLOAT,
        unit VARCHAR,
        is_approximate BOOLEAN DEFAULT FALSE,
        modifier VARCHAR,
        parent_path VARCHAR,
        time_reference VARCHAR,
        context TEXT
    );

    CREATE TABLE IF NOT EXISTS truths (
        truth_id VARCHAR PRIMARY KEY,
        request_id VARCHAR REFERENCES requests(request_id),
        field_name VARCHAR,
        field_value FLOAT,
        unit VARCHAR,
        parent_path VARCHAR,
        time_reference VARCHAR,
        source VARCHAR
    );

    CREATE TABLE IF NOT EXISTS results (
        result_id VARCHAR PRIMARY KEY,
        claim_id VARCHAR REFERENCES claims(claim_id),
        truth_id VARCHAR REFERENCES truths(truth_id),
        claimed_value FLOAT,
        true_value FLOAT,
        deviation FLOAT,
        match_status VARCHAR,
        reason TEXT,
        explanation TEXT,
        judged_at TIMESTAMP DEFAULT NOW()
    );
            
    CREATE TABLE IF NOT EXISTS kosis_stat_catalog (
        stat_id VARCHAR PRIMARY KEY,
        stat_name VARCHAR,
        org_id VARCHAR,
        org_name VARCHAR,
        category_path VARCHAR,
        keywords TEXT[],
        embedding vector(1024),
        raw_meta_json JSONB,
        fetched_at TIMESTAMP DEFAULT NOW()
    );
""")

conn.commit()
cur.close()
conn.close()

print("테이블 5개 생성 완료")