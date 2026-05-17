import re

PATTERNS: dict[str, re.Pattern[str]] = {
    "undang_undang": re.compile(r"UU\s*(?:No\.?)?\s*\d+\s*Tahun\s*\d{4}", re.IGNORECASE),
    "pasal": re.compile(r"Pasal\s+\d+(?:\s+ayat\s+\(\d+\))?", re.IGNORECASE),
    "kuhp_kuhper": re.compile(r"\b(?:KUHP|KUHPerdata|KUHAP|KUHD)\b"),
    "putusan": re.compile(
        r"Putusan\s+(?:MA|MK|PN|PA)[^\.\n]{0,80}No\.?\s*[\w/\.\-]+", re.IGNORECASE
    ),
    "perpu": re.compile(r"Perpu\s*(?:No\.?)?\s*\d+\s*Tahun\s*\d{4}", re.IGNORECASE),
    "permen": re.compile(
        r"Permen[a-zA-Z]+\s*(?:No\.?)?\s*\d+\s*Tahun\s*\d{4}", re.IGNORECASE
    ),
}


def parse(text: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for label, pattern in PATTERNS.items():
        matches = list({m.group(0).strip() for m in pattern.finditer(text)})
        if matches:
            found[label] = matches
    return found


def flatten(citations: dict[str, list[str]]) -> list[str]:
    return [m for matches in citations.values() for m in matches]
