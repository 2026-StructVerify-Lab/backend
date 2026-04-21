"""
storage/raw_storage.py — 원본 파일 저장 (S3 / MinIO)

원본 PDF, HTML, DOCX 파일을 절대 수정/삭제하지 않고 보존한다.
"""
from __future__ import annotations
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


class RawStorage:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        # TODO: boto3 S3 client 또는 MinIO client 초기화

    async def save(self, doc_id: str, data: bytes, filename: str) -> str:
        """
        원본 파일을 S3/MinIO에 저장하고 key를 반환한다.
        TODO: s3_client.put_object(Bucket=bucket, Key=key, Body=data) 구현
        """
        key = f"raw/{doc_id}/{filename}"
        logger.warning(f"Raw Storage 저장 stub: {key}")
        return key

    async def get(self, key: str) -> bytes:
        """TODO: s3_client.get_object() 구현"""
        logger.warning(f"Raw Storage 조회 stub: {key}")
        return b""
