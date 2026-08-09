#!/usr/bin/env python3
"""Build the complete vector baseline. Run audit first, then extraction/chunking/indexing."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from codefest.audit import audit
from codefest.core import detect_language, normalized_rel, phenomenon, stable_doc_id, write_jsonl
from codefest.extract import extract
from codefest.chunking import chunk_text
from codefest.vector import Encoder, build_index
from codefest.cache import ExtractionCache

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--corpus-root",type=Path,required=True)
    p.add_argument("--model",default="intfloat/multilingual-e5-base")
    p.add_argument("--device",choices=["auto","cuda","cpu"],default="auto")
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--no-fp16",action="store_true")
    p.add_argument("--target-tokens",type=int,default=360)
    p.add_argument("--max-tokens",type=int,default=480)
    p.add_argument("--enable-ocr",action="store_true")
    p.add_argument("--audit-only",action="store_true")
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--no-embeddings",action="store_true")
    p.add_argument("--cache-dir",type=Path,default=Path(".cache"))
    p.add_argument("--output-dir",type=Path,default=Path("base_vectorial/encoder_multilingual_e5_base"))
    a=p.parse_args()
    root=a.corpus_root.resolve(); report=Path("reports"); manifest=Path("data/processed/documents_manifest.jsonl")
    inventory=audit(root,report,manifest); print(f"Inventario: {inventory['total_documentos']} documentos")
    if a.audit_only: return
    if a.dry_run or a.no_embeddings:
        from codefest.dry_run import run_dry_run
        result=run_dry_run(root,a.model,a.target_tokens,a.max_tokens,Path("reports"))
        print(json.dumps(result["summary"],ensure_ascii=False,indent=2)); return
    encoder=Encoder(a.model,device=a.device,use_fp16=not a.no_fp16,cache_dir=a.cache_dir)
    extraction_cache=ExtractionCache(a.cache_dir); chunks=[]; failures=[]; cache_hits=0
    sources=sorted(x for x in root.rglob("*") if x.is_file() and x.name != ".DS_Store")
    print(f"Procesando {len(sources)} archivos...")
    for source_number, source in enumerate(sources,1):
        rel=normalized_rel(source,root); doc_id=stable_doc_id(rel)
        if phenomenon(rel) is None:
            failures.append({"ruta_relativa":rel,"error":"Excluido: sin fenomeno obligatorio"}); continue
        try:
            extracted, hit=extraction_cache.get(source,rel)
            if extracted is None:
                extracted=extract(source,enable_ocr=a.enable_ocr); extraction_cache.put(source,rel,extracted)
            else: cache_hits+=1
            position=0; pending=[]; pending_tokens=0; pending_metadata=None
            def emit_pending():
                nonlocal position,pending,pending_tokens,pending_metadata
                if not pending: return
                body=" ".join(pending)
                chunks.append({"doc_id":doc_id,"chunk_id":f"{doc_id}-chunk-{position:04d}","fuente":rel,"formato":source.suffix.lower().lstrip("."),"fenomeno":phenomenon(rel),"posicion":position,"num_tokens":pending_tokens,"texto":body,"idioma":detect_language(body),"ruta_relativa":rel,**pending_metadata})
                position+=1; pending=[]; pending_tokens=0; pending_metadata=None
            for block in extracted["blocks"]:
                if not block["text"]: continue
                try:
                    for body in chunk_text(block["text"],encoder.tokenizer,a.target_tokens,a.max_tokens):
                        tokens=len(encoder.tokenizer.encode(body,add_special_tokens=False))
                        if pending and pending_tokens+tokens>a.target_tokens: emit_pending()
                        pending.append(body); pending_tokens+=tokens
                        if pending_metadata is None: pending_metadata=block["metadata"]
                except Exception as exc:
                    failures.append({"ruta_relativa":rel,"scope":"block","metadata":block["metadata"],"error":str(exc)})
            emit_pending()
        except Exception as exc: failures.append({"ruta_relativa":rel,"error":str(exc)})
        if source_number % 100 == 0 or source_number == len(sources):
            print(f"Progreso: {source_number}/{len(sources)} archivos; chunks={len(chunks)}; fallas={len(failures)}")
    Path("data/processed").mkdir(parents=True,exist_ok=True); write_jsonl(Path("data/processed/extraction_failures.jsonl"),failures)
    if not chunks: raise RuntimeError("No se extrajeron chunks; revise extraction_failures.jsonl")
    config=build_index(chunks,a.output_dir,encoder,a.batch_size)
    (report/"build_summary.json").write_text(json.dumps({"chunks":len(chunks),"failures":len(failures),"extraction_cache_hits":cache_hits,"encoder":config},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Indice creado: {len(chunks)} chunks; fallas: {len(failures)}; cache de extracción: {cache_hits}")

if __name__=="__main__": main()
