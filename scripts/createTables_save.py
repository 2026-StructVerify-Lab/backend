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

cur = conn.cursor()

cur.execute("""
    CREATE EXTENSION IF NOT EXISTS vector;

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
""")

conn.commit()
cur.close()
conn.close()

print("테이블 4개 생성 완료")