from contextlib import asynccontextmanager

from fastapi import FastAPI
from scalar_fastapi import get_scalar_api_reference

from app.core.db import init_db
from app.router import documents, search, uploads
from app.tools import embeddings


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Eagerly load the sentence-transformers model so the first request doesn't pay the
    # multi-minute download/load cost. First-ever boot downloads ~1.3GB; subsequent boots
    # hit the HuggingFace cache.
    embeddings.encode_one("warmup")
    yield


app = FastAPI(
    title="Indonesian Legal Document Analyzer RAG",
    description="RAG engine khusus dokumen hukum berbahasa Indonesia (perjanjian, MoU, peraturan).",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(documents.router)
app.include_router(uploads.router)
app.include_router(search.router)


@app.get("/")
def root() -> dict:
    return {"status": "ok", "service": "indo-legal-rag"}


@app.get("/scalar", include_in_schema=False)
def scalar() -> object:
    return get_scalar_api_reference(openapi_url=app.openapi_url, title=app.title)
