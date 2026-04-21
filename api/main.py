"""
api/main.py — REST API 엔드포인트 (FastAPI)

/verify/text, /verify/document 엔드포인트 제공
"""
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from structverify.core.pipeline import VerificationPipeline
from structverify.core.config_loader import load_config

app = FastAPI(title="StructVerify Lab API", version="0.2.0")
config = load_config()


class VerifyTextRequest(BaseModel):
    text: str
    source_type: str = "text"


@app.post("/verify/text")
async def verify_text(req: VerifyTextRequest):
    """텍스트 기반 검증 API"""
    pipeline = VerificationPipeline(config)
    report = await pipeline.run(req.text, req.source_type)
    return report.model_dump()


@app.post("/verify/document")
async def verify_document(file: UploadFile = File(...)):
    """문서 파일 기반 검증 API"""
    # TODO: 파일 저장 → pipeline.run(filepath, source_type) 구현
    return {"status": "not_implemented"}


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.2.0"}
