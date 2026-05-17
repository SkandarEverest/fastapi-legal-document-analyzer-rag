from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.settings import settings

EMBED_DIM = 256


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    model = SentenceTransformer(settings.EMBED_MODEL)
    # sentence-transformers >=3.x renamed the method
    get_dim = getattr(model, "get_embedding_dimension", None) or getattr(
        model, "get_sentence_embedding_dimension", None
    )
    dim = get_dim() if get_dim else None
    global EMBED_DIM
    if dim is not None and dim != EMBED_DIM:
        # update the module-level constant so Chroma collection is created with the right dim
        EMBED_DIM = dim
    return model


def encode(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vecs = _model().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vecs.tolist()


def encode_one(text: str) -> list[float]:
    return encode([text])[0]
