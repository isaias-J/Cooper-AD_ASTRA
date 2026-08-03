from __future__ import annotations

import collections
import json
import re
from pathlib import Path

from .chunking import sentences, word_count
from .core import SUPPORTED, normalized_rel, phenomenon, stable_doc_id
from .extract import extract

MAX_EXAMPLES = 10
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".avif"}

def _preview(text: str, limit: int = 420) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]

def _is_list_or_table(text: str, extension: str) -> bool:
    lines=[line.strip() for line in text.splitlines() if line.strip()]
    bullets=sum(bool(re.match(r"^(?:[-*•]|\d+[.)])\s+", line)) for line in lines)
    fields=sum(line.count(":") + line.count(";") for line in lines)
    return extension in {".csv", ".xlsx", ".pbf"} or bullets >= 2 or (len(lines) >= 4 and fields >= len(lines))

def _category_for_long_unit(text: str, extension: str) -> str:
    if _is_list_or_table(text, extension): return "lista_o_tabla_texto_plano"
    if len(sentences(text)) == 1: return "oracion_aparentemente_unica_gigante"
    return "bloque_mas_de_480_tokens"

def _add_example(bucket: dict, category: str, rel: str, metadata: dict, text: str, tokens: int, words: int) -> None:
    examples=bucket.setdefault(category, [])
    if len(examples) < MAX_EXAMPLES:
        examples.append({"fuente":rel,"pagina_inicio":metadata.get("page_start"),"pagina_fin":metadata.get("page_end"),"palabras":words,"tokens":tokens,"vista_previa":_preview(text)})

def _current_coverage(index_dir: Path) -> dict:
    metadata=index_dir / "metadata.jsonl"
    if not metadata.exists(): return {"available":False}
    docs=set(); chunks=0
    with metadata.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                chunks += 1; docs.add(json.loads(line)["doc_id"])
    return {"available":True,"chunks":chunks,"documents_with_chunks":len(docs)}

def run_dry_run(root: Path, model: str, target_tokens: int, max_tokens: int, report_dir: Path) -> dict:
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model)
    report_dir.mkdir(parents=True,exist_ok=True)
    examples={}; categories=collections.Counter(); document_state={}; valid_chunks=0; valid_docs=0
    all_files=sorted(path for path in root.rglob("*") if path.is_file() and path.name != ".DS_Store")
    for source in all_files:
        rel=normalized_rel(source,root); ext=source.suffix.lower(); doc_id=stable_doc_id(rel)
        state={"doc_id":doc_id,"fuente":rel,"valid_units":0,"omitted_units":0,"categories":collections.Counter()}
        if phenomenon(rel) is None:
            category="fenomeno_faltante"; categories[category]+=1; state["categories"][category]+=1; _add_example(examples,category,rel,{},"",0,0); document_state[doc_id]=state; continue
        if ext not in SUPPORTED:
            category="formato_no_soportado"; categories[category]+=1; state["categories"][category]+=1; _add_example(examples,category,rel,{},"",0,0); document_state[doc_id]=state; continue
        if ext in IMAGE_EXTENSIONS:
            category="imagen_sin_ocr"; categories[category]+=1; state["categories"][category]+=1; _add_example(examples,category,rel,{},"",0,0); document_state[doc_id]=state; continue
        try:
            extracted=extract(source,enable_ocr=False)
        except Exception as exc:
            category="error_extraccion"; categories[category]+=1; state["categories"][category]+=1; _add_example(examples,category,rel,{},str(exc),0,0); document_state[doc_id]=state; continue
        for block in extracted.get("blocks",[]):
            text=block.get("text",""); metadata=block.get("metadata",{})
            if not text.strip():
                category="pagina_o_archivo_sin_texto"; categories[category]+=1; state["omitted_units"]+=1; state["categories"][category]+=1; _add_example(examples,category,rel,metadata,text,0,0); continue
            units=sentences(text)
            if not units:
                category="pagina_o_archivo_sin_texto"; categories[category]+=1; state["omitted_units"]+=1; state["categories"][category]+=1; _add_example(examples,category,rel,metadata,text,0,0); continue
            current_tokens=0; current_units=0
            for unit in units:
                tokens=len(tokenizer.encode(unit,add_special_tokens=False)); words=word_count(unit)
                if words>250 and tokens<=max_tokens:
                    categories["mas_de_250_palabras_dentro_de_480_tokens"]+=1; _add_example(examples,"mas_de_250_palabras_dentro_de_480_tokens",rel,metadata,unit,tokens,words)
                if tokens>max_tokens:
                    category=_category_for_long_unit(unit,ext); categories[category]+=1; state["omitted_units"]+=1; state["categories"][category]+=1; _add_example(examples,category,rel,metadata,unit,tokens,words); continue
                if current_units and current_tokens+tokens>target_tokens:
                    valid_chunks+=1; current_tokens=0; current_units=0
                current_tokens+=tokens; current_units+=1; state["valid_units"]+=1
            if current_units: valid_chunks+=1
        if state["valid_units"]: valid_docs+=1
        document_state[doc_id]=state
    complete_without=[s for s in document_state.values() if not s["valid_units"]]
    partial=[s for s in document_state.values() if s["valid_units"] and s["omitted_units"]]
    current=_current_coverage(Path("base_vectorial") / "encoder_multilingual_e5_base")
    summary={"files_scanned":len(all_files),"current_index":current,"projected":{"documents_with_chunks":valid_docs,"documents_without_chunks":len(complete_without),"partially_covered_documents":len(partial),"estimated_chunks":valid_chunks},"categories":dict(categories)}
    payload={"summary":summary,"examples":examples,"documents_without_chunks":complete_without,"partially_covered_documents":partial,"omitted_blocks_in_covered_documents":[s for s in partial if s["omitted_units"]]}
    (report_dir / "coverage_dry_run.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# Auditoría de cobertura sin embeddings","",f"Archivos revisados: **{len(all_files)}**","", "## Cobertura", "", f"- Índice actual: {current}",f"- Proyección sin límite de 250 palabras en índice: {summary['projected']}","", "## Categorías",""]
    lines += [f"- {name}: {count}" for name,count in sorted(categories.items())]
    lines += ["", "## Ejemplos representativos",""]
    for category, values in examples.items():
        lines += [f"### {category}",""]
        for value in values: lines.append(f"- `{value['fuente']}` | páginas {value['pagina_inicio']}-{value['pagina_fin']} | {value['palabras']} palabras | {value['tokens']} tokens | {value['vista_previa']}")
        lines.append("")
    (report_dir / "coverage_dry_run.md").write_text("\n".join(lines),encoding="utf-8")
    return payload
