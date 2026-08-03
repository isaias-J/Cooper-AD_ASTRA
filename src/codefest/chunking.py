from __future__ import annotations
import re
from .core import clean_text

SENTENCE_RE=re.compile(r"(?<=[.!?…])(?:[\]\)\"'»”]*)\s+(?=[¿¡]?[A-ZÁÉÍÓÚÜÑÀ-Ý])")
PARAGRAPH_RE=re.compile(r"\n\s*\n+")
LIST_ITEM_RE=re.compile(r"^\s*(?:[-*•▪◦]|\(?\d{1,3}[.)]|[A-Za-z][.)])\s+")

def _list_units(block: str) -> list[str] | None:
    """Return only complete list items; wrapped lines remain attached to their item."""
    lines=[line.strip() for line in block.splitlines() if line.strip()]
    if not lines or not any(LIST_ITEM_RE.match(line) for line in lines): return None
    items=[]; current=[]
    for line in lines:
        if LIST_ITEM_RE.match(line):
            if current: items.append(" ".join(current))
            current=[line]
        elif current:
            current.append(line)
        else:
            return None
    if current: items.append(" ".join(current))
    return items if items else None

def sentences(text: str) -> list[str]:
    """Keep complete sentences and complete structural blocks (lists/tables/headings)."""
    output=[]
    for block in PARAGRAPH_RE.split(clean_text(text)):
        listed=_list_units(block)
        if listed is not None:
            output.extend(listed)
        else:
            # A semicolon-delimited table row is one semantic unit and is never split here.
            if block.count(";") >= 2 and all(":" in piece for piece in block.split(";") if piece.strip()):
                output.append(block.strip())
            else:
                output.extend(sentence.strip() for sentence in SENTENCE_RE.split(block) if sentence.strip())
    return output
def word_count(text:str)->int: return len(re.findall(r"\S+",text))
def chunk_text(text: str, tokenizer, target_tokens=360, max_tokens=480, max_words=None) -> list[str]:
    out=[]; current=[]; count=0
    for s in sentences(text):
        n=len(tokenizer.encode(s,add_special_tokens=False))
        if n>max_tokens: raise ValueError("Una oracion excede max_tokens: requiere politica explicita, no corte automatico")
        if max_words is not None and word_count(s)>max_words: raise ValueError("Una oracion o bloque estructural excede max_words: requiere segmentacion explicita, no corte automatico")
        if current and (count+n>target_tokens or (max_words is not None and word_count(" ".join(current+[s]))>max_words)): out.append(" ".join(current)); current=[]; count=0
        current.append(s); count+=n
    if current: out.append(" ".join(current))
    return out
