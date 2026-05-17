# Indonesian Legal Document Analyzer RAG

Assignment #6 — a RAG engine specialized for Indonesian legal documents (perjanjian kerja, perjanjian sewa, MoU, NDA, peraturan), built with FastAPI and wrapped as an MCP server.

## What makes it "legal Indonesian"

1. **Indonesian sentence embeddings** — [`denaya/indoSBERT-large`](https://huggingface.co/denaya/indoSBERT-large) (base: `indobert-large-p1`, 256-dim). Materially better recall on Bahasa Indonesia legal prose than generic multilingual embeddings.
2. **Indonesian legal citation parser** — regex utility in [app/tools/id_citations.py](app/tools/id_citations.py) extracts citations in Indonesian format (`UU No. 13 Tahun 2003`, `Pasal 1320 KUHPerdata`, `Putusan MA No. ...`, `Permenaker No. 5 Tahun 2021`, etc.).
3. **Structured legal metadata** — on upload, an LLM extracts `jenis`, `para_pihak`, `tanggal_efektif`, `tanggal_berakhir`, `hukum_berlaku`, `ringkasan` via JSON-mode (see [app/tools/legal_extract.py](app/tools/legal_extract.py)).
4. **Bahasa Indonesia RAG prompt** — the search endpoint answers in Bahasa, cites sources, and appends a non-legal-advice disclaimer.

## Tech Stack

- Python 3.13+ / FastAPI / `uv`
- ChromaDB (persistent local vector store)
- OpenRouter via `openai` SDK (LLM)
- `sentence-transformers` — indoSBERT-large embeddings (local)
- SQLModel + SQLite (document/chunk metadata)
- Chonkie (recursive chunking) + pypdf
- `mcp[cli]` — MCP server (stdio transport)
- Scalar — API docs UI

## Setup

```bash
uv sync
cp .env.example .env   # fill in OPENROUTER_API_KEY
```

## Run the API

```bash
make run
# or: uv run uvicorn app.main:app --reload
```

**First boot** downloads `denaya/indoSBERT-large` from HuggingFace (~1.3GB, base is `indobert-large-p1`) and warms it into memory before serving requests. Expect the first `make run` to take several minutes; subsequent boots use the cached model and warm up in a few seconds. The same warmup runs at MCP server startup.

API docs: <http://localhost:8000/scalar> or <http://localhost:8000/docs>

## Endpoints

| Method | Path | Behavior |
| --- | --- | --- |
| `GET` | `/documents/` | List documents. Filter via `?jenis=perjanjian_kerja&pihak=PT+ABC`. |
| `GET` | `/documents/{id}` | One document with metadata + chunks (text + per-chunk citations). |
| `POST` | `/upload/` | Multipart PDF upload. Returns the created Document with extracted legal metadata. |
| `GET` | `/search?q=...&k=30&doc_ids=...` | Semantic search → top-k chunks → LLM-generated answer in Bahasa with citations. `doc_ids` (repeatable) pins retrieval to specific documents and splits k evenly across them — use for cross-document comparison. |
| `GET` | `/scalar` | API docs (Scalar). |

### Example — single document Q&A

Two sample PDFs live in [samples/](samples/):
- `Surat-PKWTT.pdf` — Indonesian unlimited-term employment agreement (PKWTT)
- `uu_no_13.pdf` — UU No. 13 Tahun 2003 tentang Ketenagakerjaan (full statute)

```bash
# Upload the sample contract
curl -F "file=@samples/Surat-PKWTT.pdf" http://localhost:8000/upload/

# List documents (should show jenis=perjanjian_kerja, para_pihak, tanggal_efektif=2026-06-01)
curl http://localhost:8000/documents/

# Semantic search — synonyms, not exact words from the doc (tests indoSBERT)
curl "http://localhost:8000/search?q=Berapa%20gaji%20bulanan%20setelah%20lulus%20probation"
curl "http://localhost:8000/search?q=Bagaimana%20cara%20saya%20resign"
curl "http://localhost:8000/search?q=Apa%20yang%20terjadi%20kalau%20ada%20bencana%20alam"
```

### Example — cross-document compliance check

Upload both a contract and the statute it should comply with, then pin retrieval to **both** documents so chunks from each are guaranteed in context.

```bash
# 1. Upload the contract and the statute
curl -F "file=@samples/Surat-PKWTT.pdf" http://localhost:8000/upload/
curl -F "file=@samples/uu_no_13.pdf" http://localhost:8000/upload/

# 2. Grab the two document IDs
curl -s http://localhost:8000/documents/ | jq -r '.[].id'
# → 01HXYZ...PKWTT
# → 01HXYZ...UU13

# 3. Cross-document query. Use doc_ids to pin both docs; bump k for the big statute.
curl "http://localhost:8000/search/?q=Apakah%20waktu%20kerja%20dalam%20perjanjian%20sudah%20sesuai%20UU&doc_ids=01HXYZ...PKWTT&doc_ids=01HXYZ...UU13&k=30"
```

For a long statute like UU 13/2003 (~150 chunks), k=30 split across 2 docs gives ~15 chunks per side — enough for the engine to pull both PKWTT Pasal 6 (42 jam/minggu) and UU Pasal 77 (max 40 jam/minggu) and flag the discrepancy. With the default `TOP_K=30` in `.env`, the `&k=` param can be omitted.

## Run the MCP server

```bash
make mcp
# or: uv run python -m app.mcp_server
```

Exposes 4 tools over stdio for Claude Desktop / other MCP clients:

- `list_documents(jenis?, pihak?)`
- `get_document(document_id)`
- `upload_document(file_path)` — local absolute path
- `search_legal_docs(query, k?)`

### Claude Desktop config snippet

```jsonc
{
  "mcpServers": {
    "indo-legal-rag": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/basic-rag-engine", "run", "python", "-m", "app.mcp_server"]
    }
  }
}
```

## Project Layout

```
app/
├── main.py             # FastAPI app
├── mcp_server.py       # MCP server (stdio)
├── core/
│   ├── settings.py     # pydantic-settings
│   └── db.py           # SQLModel engine
├── models/
│   ├── document.py     # Document table
│   └── chunk.py        # Chunk table (Chroma holds embeddings)
├── router/
│   ├── documents.py    # GET /documents/, GET /documents/{id}
│   ├── uploads.py      # POST /upload/
│   └── search.py       # GET /search
├── services/
│   └── documents.py    # Business logic shared by routers + MCP
├── tools/
│   ├── pdf.py          # pypdf text extraction
│   ├── chunking.py     # Chonkie RecursiveChunker
│   ├── embeddings.py   # indoSBERT-large wrapper
│   ├── vectorstore.py  # Chroma persistent client
│   ├── id_citations.py # Indonesian legal citation regex
│   ├── legal_extract.py# LLM JSON-mode metadata extraction
│   └── rag.py          # Retrieve + LLM answer with citations
└── utils/ids.py        # ULID helpers
```
