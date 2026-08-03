#!/usr/bin/env python3
"""Exercise every indexable extractor before expensive embedding generation."""
from __future__ import annotations
import argparse, collections, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from codefest.core import normalized_rel, phenomenon
from codefest.extract import extract

def main():
 p=argparse.ArgumentParser(); p.add_argument("--corpus-root",type=Path,required=True); p.add_argument("--enable-ocr",action="store_true"); p.add_argument("--output",type=Path,default=Path("reports/extraction_audit.json")); a=p.parse_args()
 counts=collections.Counter(); failures=[]; extracted=0; blocks=0
 for file_number,path in enumerate(sorted(x for x in a.corpus_root.rglob("*") if x.is_file() and x.name != ".DS_Store"), 1):
  if file_number % 100 == 0: print(f"Procesados: {file_number}", flush=True)
  rel=normalized_rel(path,a.corpus_root); ext=path.suffix.lower()
  if phenomenon(rel) is None: continue
  try:
   result=extract(path,enable_ocr=a.enable_ocr); n=len(result["blocks"])
   if not n: raise ValueError("Extraccion sin bloques")
   extracted+=1; blocks+=n; counts[ext]+=1
  except Exception as error: failures.append({"ruta_relativa":rel,"formato":ext,"error":str(error)})
 output={"documentos_extraidos":extracted,"bloques_extraidos":blocks,"por_formato":dict(counts),"fallas":failures,"total_fallas":len(failures)}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps({k:v for k,v in output.items() if k!="fallas"},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
