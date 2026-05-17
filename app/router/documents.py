from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.core.db import get_session
from app.services import documents as docs_service

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("/")
def list_all(
    jenis: Optional[str] = None,
    pihak: Optional[str] = None,
    session: Session = Depends(get_session),
) -> list[dict]:
    return docs_service.list_documents(session, jenis=jenis, pihak=pihak)


@router.get("/{document_id}")
def get_one(document_id: str, session: Session = Depends(get_session)) -> dict:
    result = docs_service.get_document(session, document_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return result
