from __future__ import annotations

import hashlib, json, re
from pathlib import Path

SUPPORTED = {".pdf", ".json", ".csv", ".xlsx", ".txt", ".md", ".html", ".htm", ".jpg", ".jpeg", ".png", ".avif", ".pbf"}

def normalized_rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()

def phenomenon(rel: str) -> int | None:
    first = rel.split("/", 1)[0].lower()
    for key, value in (("f1_", 1), ("f2_", 2), ("f3_", 3)):
        if first.startswith(key): return value
    return None

def detect_language(text: str) -> str:
    """Conservative ES/EN/PT detector without a network/download dependency."""
    words = re.findall(r"[a-záéíóúüñçãõ]+", text.lower())
    scores = {
        "es": sum(w in {"de","la","el","los","las","para","que","en","con","por","una","del","como"} for w in words),
        "en": sum(w in {"the","and","of","to","in","for","with","that","from","is","are","this"} for w in words),
        "pt": sum(w in {"a","é","de","da","do","para","que","em","com","por","uma","dos","das","não","também","comissão"} for w in words),
    }
    language, score = max(scores.items(), key=lambda item: item[1])
    return language if score >= 2 else "und"

def stable_doc_id(rel: str) -> str:
    # Hash only the normalized relative route: IDs survive copies of the corpus.
    return "DOC-" + hashlib.sha256(rel.encode("utf-8")).hexdigest()[:16].upper()

def clean_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)
    return text.strip()

def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)
