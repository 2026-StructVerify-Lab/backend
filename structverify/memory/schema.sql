-- [이수민 - 2026-05-14]
-- exemplar_store용 PostgreSQL 테이블 정의 (pgvector 사용)
-- 적용 방법:
--   docker exec -it pg-structverify psql -U postgres -d structverify -f schema.sql
-- 또는 scripts/init_exemplar_tables.py로 자동 실행 (다음 단계 구현)
--
-- 전제: CREATE EXTENSION vector; 이미 활성화됨 (KOSIS catalog가 사용 중)

-- ────────────────────────────────────────────────────────────────────────────
-- domain_examples — 도메인 분류 사례
-- ────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS domain_examples (
    example_id   TEXT PRIMARY KEY,
    text         TEXT NOT NULL,
    embedding    vector(1024),                  -- HCX-emb v2 차원
    domain       TEXT NOT NULL,                 -- employment, environment, population 등
    confidence   FLOAT DEFAULT 1.0,
    created_at   TIMESTAMPTZ DEFAULT now(),
    source_doc   TEXT                            -- doc_id 추적용
);

-- 임베딩 유사도 검색용 인덱스 (cosine distance)
CREATE INDEX IF NOT EXISTS idx_domain_examples_embedding
    ON domain_examples
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

-- 도메인별 통계 조회용
CREATE INDEX IF NOT EXISTS idx_domain_examples_domain
    ON domain_examples (domain);


-- ────────────────────────────────────────────────────────────────────────────
-- temporal_examples — 시간 표현 풀이 사례
-- ────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS temporal_examples (
    example_id   TEXT PRIMARY KEY,
    expression   TEXT NOT NULL,                 -- "작년", "재작년 같은 기간"
    context      TEXT,                          -- 등장 문장 + 앞뒤 문맥
    embedding    vector(1024),
    anchor_year  INTEGER NOT NULL,              -- 그 문서의 기준 연도
    resolved     TEXT NOT NULL,                 -- "2024", "2022-01..2022-11"
    basis        TEXT,                          -- 풀이 근거 (디버깅용)
    created_at   TIMESTAMPTZ DEFAULT now(),
    source_doc   TEXT
);

CREATE INDEX IF NOT EXISTS idx_temporal_examples_embedding
    ON temporal_examples
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

-- anchor_year 가까운 사례 우대 정렬용
CREATE INDEX IF NOT EXISTS idx_temporal_examples_anchor
    ON temporal_examples (anchor_year);

-- expression 정확 매칭 빠른 lookup용 (선택)
CREATE INDEX IF NOT EXISTS idx_temporal_examples_expression
    ON temporal_examples (expression);
