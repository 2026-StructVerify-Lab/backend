import os
import json
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from psycopg2.extensions import register_adapter
from psycopg2.extras import Json

VECTOR_DIM = 1024

def normalize_keywords(value):
    if value is None:
        return []

    if isinstance(value, list):
        return [str(v) for v in value]

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            pass

        return [v.strip() for v in value.split(",") if v.strip()]

    return [str(value)]

def to_pgvector(vec) -> str | None:
    if vec is None:
        return None

    if isinstance(vec, str):
        # 이미 "[...]" 형태면 그대로 사용
        if vec.strip().startswith("["):
            return vec.strip()
        try:
            vec = json.loads(vec)
        except json.JSONDecodeError:
            return None

    if not isinstance(vec, list):
        return None

    if len(vec) != VECTOR_DIM:
        print(f"[WARN] embedding dim mismatch: {len(vec)} != {VECTOR_DIM}")
        return None

    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def get_embedding(item: dict):
    emb = (
        item.get("embedding")
        or item.get("embeddings")
        or item.get("embedding_vector")
        or item.get("vector")
        or item.get("values")
    )

    if isinstance(emb, dict):
        emb = emb.get("embedding") or emb.get("vector") or emb.get("values")

    return emb


def load_json_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["data", "items", "rows", "result", "catalog"]:
            if isinstance(data.get(key), list):
                return data[key]

    return []


def main():
    load_dotenv()

    json_path = Path(os.getenv("KOSIS_CATALOG_JSON", "kosis_catalog_cache.json"))

    if not json_path.exists():
        raise FileNotFoundError(f"JSON 파일 없음: {json_path.resolve()}")

    rows = load_json_rows(json_path)
    if not rows:
        raise ValueError("JSON 안에 적재할 row가 없음")

    sample = rows[0]
    print("[DEBUG] sample keys:", sample.keys())

    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "structverify"),
        user=os.getenv("POSTGRES_USER", "structverify"),
        password=os.getenv("POSTGRES_PASSWORD", "svpass123"),
    )

    cur = conn.cursor()

    cur.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
    if not cur.fetchone()[0]:
        raise RuntimeError("현재 DB에 pgvector extension이 없음. CREATE EXTENSION vector 먼저 실행 필요")

    inserted = 0
    skipped = 0
    embedding_inserted = 0

    for item in rows:
        stat_id = item.get("stat_id") or item.get("tbl_id") or item.get("TBL_ID")
        stat_name = item.get("stat_name") or item.get("tbl_nm") or item.get("TBL_NM")
        org_id = item.get("org_id") or item.get("ORG_ID")
        org_name = item.get("org_name") or item.get("ORG_NM")
        category_path = item.get("category_path") or ""
        keywords = normalize_keywords(item.get("keywords"))
        if not stat_id:
            skipped += 1
            continue

        embedding = to_pgvector(get_embedding(item))
        if embedding is not None:
            embedding_inserted += 1

        cur.execute(
            """
            INSERT INTO kosis_stat_catalog (
                stat_id,
                stat_name,
                org_id,
                org_name,
                category_path,
                keywords,
                embedding,
                raw_meta_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s::jsonb)
            ON CONFLICT (stat_id) DO UPDATE
            SET stat_name = EXCLUDED.stat_name,
                org_id = EXCLUDED.org_id,
                org_name = EXCLUDED.org_name,
                category_path = EXCLUDED.category_path,
                keywords = EXCLUDED.keywords,
                embedding = EXCLUDED.embedding,
                raw_meta_json = EXCLUDED.raw_meta_json,
                fetched_at = NOW()
            """,
            (
                stat_id,
                stat_name,
                org_id,
                org_name,
                category_path,
                keywords,
                embedding,
                json.dumps(item, ensure_ascii=False),
            ),
        )

        inserted += 1

        if inserted % 1000 == 0:
            conn.commit()
            print(f"[LOAD] {inserted}/{len(rows)} rows upserted, skipped={skipped}")

    conn.commit()

    cur.execute("SELECT COUNT(*) FROM kosis_stat_catalog")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM kosis_stat_catalog WHERE embedding IS NOT NULL")
    embedding_total = cur.fetchone()[0]

    cur.close()
    conn.close()

    print(f"[DONE] inserted/upserted={inserted}, skipped={skipped}")
    print(f"[DONE] inserted_with_embedding={embedding_inserted}")
    print(f"[DONE] total={total}, embedding_rows={embedding_total}")


if __name__ == "__main__":
    main()