import json
import shutil
from pathlib import Path
from typing import Optional

from sqlmodel import Session, select

from app.core.settings import settings
from app.models.chunk import Chunk
from app.models.document import Document
from app.tools import chunking, embeddings, id_citations, legal_extract, pdf, vectorstore
from app.utils.ids import new_id


def _to_dict(doc: Document, chunks: Optional[list[Chunk]] = None) -> dict:
    data = {
        "id": doc.id,
        "filename": doc.filename,
        "jenis": doc.jenis,
        "para_pihak": json.loads(doc.para_pihak_json) if doc.para_pihak_json else [],
        "tanggal_efektif": doc.tanggal_efektif.isoformat() if doc.tanggal_efektif else None,
        "tanggal_berakhir": doc.tanggal_berakhir.isoformat() if doc.tanggal_berakhir else None,
        "hukum_berlaku": doc.hukum_berlaku,
        "ringkasan": doc.ringkasan,
        "uploaded_at": doc.uploaded_at.isoformat(),
    }
    if chunks is not None:
        data["chunks"] = [
            {
                "id": c.id,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "citations": json.loads(c.citations_json) if c.citations_json else {},
            }
            for c in chunks
        ]
    return data


def list_documents(
    session: Session, jenis: Optional[str] = None, pihak: Optional[str] = None
) -> list[dict]:
    stmt = select(Document)
    if jenis:
        stmt = stmt.where(Document.jenis == jenis)
    docs = session.exec(stmt).all()
    out = [_to_dict(d) for d in docs]
    if pihak:
        needle = pihak.lower()
        out = [d for d in out if any(needle in p.lower() for p in d["para_pihak"])]
    return out


def get_document(session: Session, document_id: str) -> Optional[dict]:
    doc = session.get(Document, document_id)
    if not doc:
        return None
    chunks = session.exec(
        select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
    ).all()
    return _to_dict(doc, list(chunks))


def upload_document(session: Session, filename: str, source_path: Path) -> dict:
    doc_id = new_id()
    stored = settings.UPLOAD_DIR / f"{doc_id}_{filename}"
    shutil.copyfile(source_path, stored)

    raw_text = pdf.extract_text(stored)
    if not raw_text.strip():
        raise ValueError("PDF appears to contain no extractable text")

    chunk_texts = chunking.chunk_text(raw_text)
    if not chunk_texts:
        raise ValueError("Chunker produced no chunks")

    vectors = embeddings.encode(chunk_texts)

    chunk_rows: list[Chunk] = []
    chunk_ids: list[str] = []
    for idx, text in enumerate(chunk_texts):
        citations = id_citations.parse(text)
        row = Chunk(
            document_id=doc_id,
            chunk_index=idx,
            text=text,
            citations_json=json.dumps(citations) if citations else None,
        )
        chunk_rows.append(row)
        chunk_ids.append(row.id)

    vectorstore.add_chunks(chunk_ids, chunk_texts, vectors, doc_id)

    metadata = legal_extract.extract(chunk_texts)

    doc = Document(
        id=doc_id,
        filename=filename,
        storage_path=str(stored),
        jenis=metadata.get("jenis"),
        para_pihak_json=json.dumps(metadata.get("para_pihak", [])),
        tanggal_efektif=metadata.get("tanggal_efektif"),
        tanggal_berakhir=metadata.get("tanggal_berakhir"),
        hukum_berlaku=metadata.get("hukum_berlaku"),
        ringkasan=metadata.get("ringkasan"),
    )

    session.add(doc)
    for row in chunk_rows:
        session.add(row)
    session.commit()
    session.refresh(doc)
    return _to_dict(doc)
