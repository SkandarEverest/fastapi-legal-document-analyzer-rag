from chonkie import RecursiveChunker

_chunker = RecursiveChunker(chunk_size=512)


def chunk_text(text: str) -> list[str]:
    if not text.strip():
        return []
    return [c.text for c in _chunker(text)]
