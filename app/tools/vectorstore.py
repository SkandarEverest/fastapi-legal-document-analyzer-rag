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


def _parse_results(res: dict) -> list[dict]:
    out = []
    for cid, text, meta, dist in zip(
        res.get("ids", [[]])[0],
        res.get("documents", [[]])[0],
        res.get("metadatas", [[]])[0],
        res.get("distances", [[]])[0],
        strict=False,
    ):
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


def query(embedding: list[float], k: int = 5, doc_ids: list[str] | None = None) -> list[dict]:
    if not doc_ids:
        res = _collection().query(query_embeddings=[embedding], n_results=k)
        return _parse_results(res)

    if len(doc_ids) == 1:
        res = _collection().query(
            query_embeddings=[embedding],
            n_results=k,
            where={"document_id": doc_ids[0]},
        )
        return _parse_results(res)

    # Multiple docs: retrieve k_each from each to guarantee representation from all.
    k_each = max(2, k // len(doc_ids))
    hits: list[dict] = []
    seen: set[str] = set()
    for doc_id in doc_ids:
        res = _collection().query(
            query_embeddings=[embedding],
            n_results=k_each,
            where={"document_id": doc_id},
        )
        for h in _parse_results(res):
            if h["chunk_id"] not in seen:
                seen.add(h["chunk_id"])
                hits.append(h)
    return sorted(hits, key=lambda h: h["distance"])


def delete_by_document(document_id: str) -> None:
    _collection().delete(where={"document_id": document_id})
