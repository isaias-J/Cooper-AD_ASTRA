#!/usr/bin/env python3
"""Build the complete vector baseline. Run audit first, then extraction/chunking/indexing."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from codefest.audit import audit
from codefest.core import detect_language, normalized_rel, phenomenon, stable_doc_id, write_jsonl
from codefest.extract import extract, remove_repeated_ocr_blocks
from codefest.chunking import chunk_text
from codefest.vector import Encoder, build_index
from codefest.cache import ExtractionCache

def main():
    pipeline_started=time.time()
    p=argparse.ArgumentParser()
    p.add_argument("--corpus-root",type=Path,required=True)
    p.add_argument("--model",default="intfloat/multilingual-e5-base")
    p.add_argument("--device",choices=["auto","cuda","cpu"],default="auto")
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--no-fp16",action="store_true")
    p.add_argument("--target-tokens",type=int,default=360)
    p.add_argument("--max-tokens",type=int,default=480)
    p.add_argument("--enable-ocr",action="store_true")
    p.add_argument("--ocr-languages",default="spa+eng+por")
    p.add_argument("--ocr-dpi",type=int,default=250)
    p.add_argument("--ocr-min-confidence",type=float,default=60.0)
    p.add_argument("--tesseract-command",type=Path,default=None)
    p.add_argument("--tessdata-dir",type=Path,default=None)
    p.add_argument("--ocr-image-allowlist",type=Path,default=None)
    p.add_argument("--audit-only",action="store_true")
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--no-embeddings",action="store_true")
    p.add_argument("--cache-dir",type=Path,default=Path(".cache"))
    p.add_argument("--output-dir",type=Path,default=Path("base_vectorial/encoder_multilingual_e5_base"))
    a=p.parse_args()
    if a.ocr_dpi < 150 or a.ocr_dpi > 400: raise ValueError("--ocr-dpi debe estar entre 150 y 400")
    if a.ocr_min_confidence < 0 or a.ocr_min_confidence > 100: raise ValueError("--ocr-min-confidence debe estar entre 0 y 100")
    root=a.corpus_root.resolve(); report=Path("reports"); manifest=Path("data/processed/documents_manifest.jsonl")
    image_allowlist=None
    if a.ocr_image_allowlist:
        image_allowlist={
            line.strip().replace("\\","/")
            for line in a.ocr_image_allowlist.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
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
        is_image=source.suffix.lower() in {".jpg",".jpeg",".png",".avif"}
        if a.enable_ocr and is_image and image_allowlist is not None and rel not in image_allowlist:
            failures.append({"ruta_relativa":rel,"error":"Excluido tras revision visual: imagen sin texto analitico relevante"}); continue
        ocr_requested=bool(a.enable_ocr and (source.suffix.lower()==".pdf" or is_image))
        cache_variant=(
            f"extract-v2|ocr=tesseract5|languages={a.ocr_languages}|dpi={a.ocr_dpi}|confidence={a.ocr_min_confidence:g}"
            if ocr_requested else "default"
        )
        try:
            extracted, hit=extraction_cache.get(source,rel,variant=cache_variant)
            if extracted is None and ocr_requested and source.suffix.lower()==".pdf":
                native, native_hit=extraction_cache.get(source,rel)
                if native is not None and native.get("blocks"):
                    extracted, hit=native, native_hit
            if extracted is None:
                extracted=extract(
                    source,
                    enable_ocr=ocr_requested,
                    ocr_languages=a.ocr_languages,
                    ocr_dpi=a.ocr_dpi,
                    tesseract_command=a.tesseract_command,
                    tessdata_dir=a.tessdata_dir,
                    ocr_min_confidence=a.ocr_min_confidence,
                )
                extraction_cache.put(source,rel,extracted,variant=cache_variant)
            else: cache_hits+=1
            extracted=remove_repeated_ocr_blocks(extracted)
            if not extracted.get("blocks"):
                failures.append({"ruta_relativa":rel,"error":"Extraccion sin bloques de texto analitico"})
            position=0; pending=[]; pending_tokens=0; pending_metadata=None
            def emit_pending():
                nonlocal position,pending,pending_tokens,pending_metadata
                if not pending: return
                body=" ".join(pending)
                chunks.append({"doc_id":doc_id,"chunk_id":f"{doc_id}-chunk-{position:04d}","fuente":rel,"formato":source.suffix.lower().lstrip("."),"fenomeno":phenomenon(rel),"posicion":position,"num_tokens":pending_tokens,"texto":body,"idioma":detect_language(body),"ruta_relativa":rel,**pending_metadata})
                position+=1; pending=[]; pending_tokens=0; pending_metadata=None
            def merge_metadata(current, incoming):
                if current is None: return dict(incoming)
                merged=dict(current)
                starts=[value for value in (current.get("page_start"),incoming.get("page_start")) if value is not None]
                ends=[value for value in (current.get("page_end"),incoming.get("page_end")) if value is not None]
                if starts: merged["page_start"]=min(starts)
                if ends: merged["page_end"]=max(ends)
                confidences=[value for value in (current.get("ocr_confidence"),incoming.get("ocr_confidence")) if value is not None]
                if confidences: merged["ocr_confidence"]=min(confidences)
                if current.get("ocr_paragraph") != incoming.get("ocr_paragraph"):
                    merged.pop("ocr_block",None); merged.pop("ocr_paragraph",None)
                return merged
            for block in extracted["blocks"]:
                if not block["text"]: continue
                try:
                    for body in chunk_text(block["text"],encoder.tokenizer,a.target_tokens,a.max_tokens):
                        tokens=len(encoder.tokenizer.encode(body,add_special_tokens=False))
                        if pending and pending_tokens+tokens>a.target_tokens: emit_pending()
                        pending.append(body); pending_tokens+=tokens
                        pending_metadata=merge_metadata(pending_metadata,block["metadata"])
                except Exception as exc:
                    failures.append({"ruta_relativa":rel,"scope":"block","metadata":block["metadata"],"error":str(exc)})
            emit_pending()
        except Exception as exc: failures.append({"ruta_relativa":rel,"error":str(exc)})
        if source_number % 100 == 0 or source_number == len(sources):
            print(f"Progreso: {source_number}/{len(sources)} archivos; chunks={len(chunks)}; fallas={len(failures)}")
    Path("data/processed").mkdir(parents=True,exist_ok=True); write_jsonl(Path("data/processed/extraction_failures.jsonl"),failures)
    if not chunks: raise RuntimeError("No se extrajeron chunks; revise extraction_failures.jsonl")
    config=build_index(chunks,a.output_dir,encoder,a.batch_size)
    config |= {
        "target_tokens": a.target_tokens,
        "max_tokens": a.max_tokens,
        "extraction_cache_hits": cache_hits,
        "pipeline_seconds": round(time.time()-pipeline_started,2),
        "ocr": {
            "enabled": a.enable_ocr,
            "engine": "tesseract-5" if a.enable_ocr else None,
            "languages": a.ocr_languages if a.enable_ocr else None,
            "dpi": a.ocr_dpi if a.enable_ocr else None,
            "minimum_confidence": a.ocr_min_confidence if a.enable_ocr else None,
            "image_allowlist": str(a.ocr_image_allowlist) if a.ocr_image_allowlist else None,
        },
    }
    (a.output_dir/"encoder_config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    (report/"build_summary.json").write_text(json.dumps({"chunks":len(chunks),"failures":len(failures),"extraction_cache_hits":cache_hits,"encoder":config},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Indice creado: {len(chunks)} chunks; fallas: {len(failures)}; cache de extracción: {cache_hits}")

if __name__=="__main__": main()
