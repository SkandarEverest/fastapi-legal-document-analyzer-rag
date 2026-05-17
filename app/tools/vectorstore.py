from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.settings import settings


@lru_cache(maxsize=1)
def _client() -> chromadb.api.ClientAPI:
    return chromadb.PersistentClient(
        path=str(settings.CHROMA_DIR),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


@lru_cache(maxsize=1)
def _collection():
    return _client().get_or_create_collection(
        name=settings.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(
    chunk_ids: list[str],
    texts: list[str],
    embeddings: list[list[float]],
    document_id: str,
) -> None:
    if not chunk_ids:
        return
    metadatas = [{"document_id": document_id, "chunk_index": i} for i in range(len(chunk_ids))]
    _collection().add(
        ids=chunk_ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def query(embedding: list[float], k: int = 5) -> list[dict]:
    res = _collection().query(query_embeddings=[embedding], n_results=k)
    out = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]
    for cid, text, meta, dist in zip(ids, docs, metas, distances, strict=False):
        out.append(
            {
                "chunk_id": cid,
                "text": text,
                "document_id": meta.get("document_id"),
                "chunk_index": meta.get("chunk_index"),
                "distance": dist,
            }
        )
    return out


def delete_by_document(document_id: str) -> None:
    _collection().delete(where={"document_id": document_id})
