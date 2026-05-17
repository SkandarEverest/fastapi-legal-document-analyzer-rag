import json

from openai import OpenAI

from app.core.settings import settings
from app.tools import embeddings, id_citations, vectorstore

SYSTEM_PROMPT = """Anda adalah analis hukum yang membantu pengguna memahami dokumen hukum Indonesia.

Aturan:
- Jawab dalam Bahasa Indonesia.
- Hanya gunakan informasi dari KONTEKS yang diberikan. Jika tidak ada di konteks, katakan tidak diketahui.
- Selalu sebut sumber (mis. dokumen, pasal, ayat) ketika mengutip.
- Akhiri jawaban dengan disclaimer singkat: "Catatan: ini bukan nasihat hukum."
"""


def _client() -> OpenAI:
    return OpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
    )


def search(query: str, k: int | None = None) -> dict:
    top_k = k or settings.TOP_K
    q_vec = embeddings.encode_one(query)
    hits = vectorstore.query(q_vec, k=top_k)

    citations = []
    context_blocks = []
    for i, h in enumerate(hits, start=1):
        refs = id_citations.flatten(id_citations.parse(h["text"]))
        citations.append(
            {
                "document_id": h["document_id"],
                "chunk_index": h["chunk_index"],
                "snippet": h["text"][:400],
                "legal_refs": refs,
                "distance": h["distance"],
            }
        )
        context_blocks.append(
            f"[Sumber {i} | dok={h['document_id']} | chunk={h['chunk_index']}]\n{h['text']}"
        )

    if not hits:
        return {
            "answer": "Tidak ada dokumen yang relevan ditemukan. Catatan: ini bukan nasihat hukum.",
            "citations": [],
        }

    if not settings.OPENROUTER_API_KEY:
        return {
            "answer": "[LLM tidak dikonfigurasi: OPENROUTER_API_KEY kosong]",
            "citations": citations,
        }

    context = "\n\n".join(context_blocks)
    try:
        resp = _client().chat.completions.create(
            model=settings.CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"PERTANYAAN: {query}\n\nKONTEKS:\n{context}",
                },
            ],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content or ""
    except Exception as e:
        answer = f"[Gagal memanggil LLM: {e}]"

    return {"answer": answer, "citations": citations}
