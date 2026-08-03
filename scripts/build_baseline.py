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

def main():
 p=argparse.ArgumentParser(); p.add_argument("--corpus-root",type=Path,required=True); p.add_argument("--model",default="intfloat/multilingual-e5-base"); p.add_argument("--target-tokens",type=int,default=360); p.add_argument("--max-tokens",type=int,default=480); p.add_argument("--batch-size",type=int,default=32); p.add_argument("--enable-ocr",action="store_true"); p.add_argument("--audit-only",action="store_true"); a=p.parse_args()
 root=a.corpus_root.resolve(); report=Path("reports"); manifest=Path("data/processed/documents_manifest.jsonl")
 inventory=audit(root,report,manifest); print(f"Inventario: {inventory['total_documentos']} documentos")
 if a.audit_only: return
 encoder=Encoder(a.model); chunks=[]; failures=[]
 for source in sorted(x for x in root.rglob("*") if x.is_file() and x.name != ".DS_Store"):
  rel=normalized_rel(source,root); doc_id=stable_doc_id(rel)
  if phenomenon(rel) is None:
   failures.append({"ruta_relativa":rel,"error":"Excluido: sin fenomeno obligatorio"}); continue
  try:
   extracted=extract(source,enable_ocr=a.enable_ocr); position=0
   for block in extracted["blocks"]:
    if not block["text"]: continue
    for body in chunk_text(block["text"],encoder.tokenizer,a.target_tokens,a.max_tokens):
     chunks.append({"doc_id":doc_id,"chunk_id":f"{doc_id}-chunk-{position:04d}","fuente":rel,"formato":source.suffix.lower().lstrip("."),"fenomeno":phenomenon(rel),"posicion":position,"num_tokens":len(encoder.tokenizer.encode(body,add_special_tokens=False)),"texto":body,"idioma":detect_language(body),"ruta_relativa":rel,**block["metadata"]}); position+=1
  except Exception as exc: failures.append({"ruta_relativa":rel,"error":str(exc)})
 Path("data/processed").mkdir(parents=True,exist_ok=True); write_jsonl(Path("data/processed/extraction_failures.jsonl"),failures)
 if not chunks: raise RuntimeError("No se extrajeron chunks; revise extraction_failures.jsonl")
 config=build_index(chunks,Path("base_vectorial/encoder_multilingual_e5_base"),encoder,a.batch_size)
 (report/"build_summary.json").write_text(json.dumps({"chunks":len(chunks),"failures":len(failures),"encoder":config},ensure_ascii=False,indent=2),encoding="utf-8")
 print(f"Indice creado: {len(chunks)} chunks; fallas: {len(failures)}")
if __name__=="__main__": main()
