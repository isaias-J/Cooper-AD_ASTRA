from __future__ import annotations
import re
from .core import clean_text

SENTENCE_RE=re.compile(r"(?<=[.!?…])(?:[\]\)\"'»”]*)\s+(?=[¿¡]?[A-ZÁÉÍÓÚÜÑÀ-Ý])")
def sentences(text: str) -> list[str]: return [x.strip() for x in SENTENCE_RE.split(clean_text(text)) if x.strip()]
def word_count(text:str)->int: return len(re.findall(r"\S+",text))
def chunk_text(text: str, tokenizer, target_tokens=360, max_tokens=480) -> list[str]:
    out=[]; current=[]; count=0
    for s in sentences(text):
        n=len(tokenizer.encode(s,add_special_tokens=False))
        if n>max_tokens: raise ValueError("Una oracion excede max_tokens: requiere politica explicita, no corte automatico")
        if current and count+n>target_tokens: out.append(" ".join(current)); current=[]; count=0
        current.append(s); count+=n
    if current: out.append(" ".join(current))
    return out
