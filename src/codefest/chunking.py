from __future__ import annotations
import re
from .core import clean_text

SENTENCE_RE=re.compile(r"(?<=[.!?…])(?:[\]\)\"'»”]*)\s+(?=[¿¡]?[A-ZÁÉÍÓÚÜÑÀ-Ý])")
SENTENCE_BOUNDARY_RE=re.compile(r"(?P<closers>[\]\)\"'»”]*)\s+(?=[¿¡]?[A-ZÁÉÍÓÚÜÑÀ-Ý])")
PARAGRAPH_RE=re.compile(r"\n\s*\n+")
LIST_ITEM_RE=re.compile(r"^\s*(?:[-*•▪◦]|\(?\d{1,3}[.)]|[A-Za-z][.)])\s+")

def _token_count(text: str, tokenizer) -> int:
    # Use the fast tokenizer backend when available to avoid warnings for oversized units.
    backend=getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        return len(backend.encode(text).ids)
    return len(tokenizer.encode(text,add_special_tokens=False))

def _split_oversized_unit(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Split long structural units without cutting sentences or table fields."""
    marker=LIST_ITEM_RE.match(text)
    if marker:
        prefix=text[:marker.end()]
        body=text[marker.end():].strip()
        parts=_sentence_parts(body)
        if len(parts) <= 1: return []
        result=[prefix+parts[0]]+[part for part in parts[1:]]
        return result if all(_token_count(part,tokenizer)<=max_tokens for part in result) else []
    if ";" not in text: return []
    fields=[field.strip() for field in text.split(";") if field.strip()]
    if len(fields) <= 1: return []
    result=[]; current=[]
    for field in fields:
        candidate="; ".join(current+[field])
        if current and _token_count(candidate,tokenizer)>max_tokens:
            result.append("; ".join(current)); current=[field]
        else:
            current.append(field)
    if current: result.append("; ".join(current))
    return result if all(_token_count(part,tokenizer)<=max_tokens for part in result) else []

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

def _sentence_parts(block: str) -> list[str]:
    """Split after terminal punctuation while retaining closing quotes/brackets."""
    output=[]; start=0
    for match in SENTENCE_BOUNDARY_RE.finditer(block):
        prefix=block[start:match.start()]
        if not prefix or prefix[-1] not in ".!?…":
            continue
        end=match.start()+len(match.group("closers"))
        output.append(block[start:end].strip()); start=match.end()
    tail=block[start:].strip()
    if tail: output.append(tail)
    return [item for item in output if item]

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
                output.extend(_sentence_parts(block))
    return output
def word_count(text:str)->int: return len(re.findall(r"\S+",text))
def chunk_text(text: str, tokenizer, target_tokens=360, max_tokens=480, max_words=None) -> list[str]:
    out=[]; current=[]; count=0
    for s in sentences(text):
        n=_token_count(s,tokenizer)
        if n>max_tokens:
            split=_split_oversized_unit(s,tokenizer,max_tokens)
            if split:
                for part in split:
                    if _token_count(part,tokenizer)>max_tokens:
                        raise ValueError("Una oracion excede max_tokens: requiere politica explicita, no corte automatico")
                for part in split:
                    if current and (count+_token_count(part,tokenizer)>target_tokens or (max_words is not None and word_count(" ".join(current+[part]))>max_words)):
                        out.append(" ".join(current)); current=[]; count=0
                    current.append(part); count+=_token_count(part,tokenizer)
                continue
            raise ValueError("Una oracion excede max_tokens: requiere politica explicita, no corte automatico")
        if max_words is not None and word_count(s)>max_words: raise ValueError("Una oracion o bloque estructural excede max_words: requiere segmentacion explicita, no corte automatico")
        if current and (count+n>target_tokens or (max_words is not None and word_count(" ".join(current+[s]))>max_words)): out.append(" ".join(current)); current=[]; count=0
        current.append(s); count+=n
    if current: out.append(" ".join(current))
    return out
