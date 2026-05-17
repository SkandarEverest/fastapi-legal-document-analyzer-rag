from fastapi import APIRouter, Query

from app.tools import rag

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/")
def search(
    q: str = Query(..., min_length=1),
    k: int | None = None,
    doc_ids: list[str] = Query(default=[]),
) -> dict:
    return rag.search(q, k=k, doc_ids=doc_ids or None)
