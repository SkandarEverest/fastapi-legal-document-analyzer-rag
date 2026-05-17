from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.settings import settings

EMBED_DIM = 256


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    model = SentenceTransformer(settings.EMBED_MODEL)
    dim = model.get_sentence_embedding_dimension()
    if dim != EMBED_DIM:
        raise RuntimeError(
            f"Embedding model {settings.EMBED_MODEL} reported dim={dim}, expected {EMBED_DIM}. "
            "Delete the chroma/ directory and update EMBED_DIM in app/tools/embeddings.py."
        )
    return model


def encode(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vecs = _model().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vecs.tolist()


def encode_one(text: str) -> list[float]:
    return encode([text])[0]
