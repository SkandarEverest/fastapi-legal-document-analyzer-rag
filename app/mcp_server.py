from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from sqlmodel import Session

from app.core.db import engine, init_db
from app.services import documents as docs_service
from app.tools import embeddings, rag

init_db()
embeddings.encode_one("warmup")  # eager-load indoSBERT so first tool call isn't slow

mcp = FastMCP("indo-legal-rag")


@mcp.tool()
def list_documents(jenis: Optional[str] = None, pihak: Optional[str] = None) -> list[dict]:
    """List all uploaded Indonesian legal documents. Optionally filter by `jenis`
    (perjanjian_kerja, perjanjian_sewa, mou, nda, perjanjian_jual_beli, peraturan, lainnya)
    or `pihak` (substring of a party name).
    """
    with Session(engine) as session:
        return docs_service.list_documents(session, jenis=jenis, pihak=pihak)


@mcp.tool()
def get_document(document_id: str) -> dict:
    """Get a single document with its metadata and chunks (text + per-chunk citations)."""
    with Session(engine) as session:
        result = docs_service.get_document(session, document_id)
        if result is None:
            return {"error": "Document not found", "document_id": document_id}
        return result


@mcp.tool()
def upload_document(file_path: str) -> dict:
    """Upload a local PDF (by absolute path) into the legal RAG. Extracts text, chunks,
    embeds with indoSBERT, parses Indonesian legal citations, and extracts metadata via LLM.
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return {"error": f"File not found: {file_path}"}
    if path.suffix.lower() != ".pdf":
        return {"error": "Only PDF files are supported"}
    with Session(engine) as session:
        try:
            return docs_service.upload_document(session, path.name, path)
        except ValueError as e:
            return {"error": str(e)}


@mcp.tool()
def search_legal_docs(query: str, k: int = 5) -> dict:
    """Semantic search across uploaded Indonesian legal documents.
    Returns `{answer, citations: [{document_id, chunk_index, snippet, legal_refs}]}`.
    """
    return rag.search(query, k=k)


if __name__ == "__main__":
    mcp.run()
