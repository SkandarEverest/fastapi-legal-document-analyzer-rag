import json
from datetime import date
from typing import Optional

import dateparser
from openai import OpenAI

from app.core.settings import settings

SYSTEM_PROMPT = """Anda adalah asisten ekstraksi metadata dokumen hukum Indonesia.
Tugas Anda: membaca cuplikan dokumen hukum berbahasa Indonesia dan mengekstrak metadata terstruktur.

Keluarkan JSON dengan field berikut (gunakan null jika tidak diketahui):
- jenis: salah satu dari ["perjanjian_kerja", "perjanjian_sewa", "mou", "nda", "perjanjian_jual_beli", "peraturan", "lainnya"]
- para_pihak: array nama-nama pihak (string), contoh ["PT ABC", "Budi Santoso"]
- tanggal_efektif: tanggal mulai berlaku dalam format string apapun yang ada di dokumen (atau null)
- tanggal_berakhir: tanggal berakhir (atau null)
- hukum_berlaku: hukum yang berlaku, mis. "Hukum Republik Indonesia" (atau null)
- ringkasan: ringkasan dokumen 2-3 kalimat dalam Bahasa Indonesia

Hanya keluarkan JSON valid, tanpa penjelasan tambahan."""


def _client() -> OpenAI:
    return OpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
    )


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    parsed = dateparser.parse(value, languages=["id", "en"])
    return parsed.date() if parsed else None


def extract(chunks: list[str]) -> dict:
    """Extract structured metadata from the first chunks of a legal document."""
    if not settings.OPENROUTER_API_KEY:
        return {
            "jenis": None,
            "para_pihak": [],
            "tanggal_efektif": None,
            "tanggal_berakhir": None,
            "hukum_berlaku": None,
            "ringkasan": None,
        }

    excerpt = "\n\n".join(chunks[:6])[:8000]

    try:
        resp = _client().chat.completions.create(
            model=settings.CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Cuplikan dokumen:\n\n{excerpt}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        return {
            "jenis": None,
            "para_pihak": [],
            "tanggal_efektif": None,
            "tanggal_berakhir": None,
            "hukum_berlaku": None,
            "ringkasan": f"[ekstraksi gagal: {e}]",
        }

    return {
        "jenis": data.get("jenis"),
        "para_pihak": data.get("para_pihak") or [],
        "tanggal_efektif": _parse_date(data.get("tanggal_efektif")),
        "tanggal_berakhir": _parse_date(data.get("tanggal_berakhir")),
        "hukum_berlaku": data.get("hukum_berlaku"),
        "ringkasan": data.get("ringkasan"),
    }
