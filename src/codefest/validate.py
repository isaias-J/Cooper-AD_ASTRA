from __future__ import annotations
import json
from pathlib import Path
from .chunking import word_count
REQUIRED={"doc_id","chunk_id","fuente","formato","fenomeno","posicion","num_tokens","texto"}
def validate_metadata(path:Path, tokenizer=None, max_tokens=480):
 seen=set(); errors=[]
 with path.open(encoding="utf-8") as stream: lines=list(stream)
 for i,line in enumerate(lines):
  x=json.loads(line); missing=REQUIRED-set(x)
  if missing: errors.append(f"{i}: faltan {missing}")
  if x.get("chunk_id") in seen: errors.append(f"{i}: chunk_id duplicado")
  seen.add(x.get("chunk_id"))
  if not x.get("texto","").strip(): errors.append(f"{i}: texto vacio")
  if not isinstance(x.get("fenomeno"),int) or x.get("fenomeno") not in {1,2,3}: errors.append(f"{i}: fenomeno invalido")
  if not isinstance(x.get("posicion"),int) or x.get("posicion") < 0: errors.append(f"{i}: posicion invalida")
  if not isinstance(x.get("num_tokens"),int) or x.get("num_tokens") <= 0 or x.get("num_tokens") > max_tokens: errors.append(f"{i}: num_tokens invalido")
  if tokenizer and len(tokenizer.encode(x["texto"],add_special_tokens=False))!=x["num_tokens"]: errors.append(f"{i}: num_tokens inconsistente")
 return errors
def validate_results(path:Path, official=True, known_chunk_ids=None, known_doc_ids=None):
 with path.open(encoding="utf-8") as stream: rows=[json.loads(line) for line in stream if line.strip()]
 errors=[]
 if official and len(rows)!=50: errors.append("Se requieren exactamente 50 lineas")
 for i,row in enumerate(rows,1):
  if official and row.get("query_id")!=f"q{i:03d}": errors.append(f"Linea {i}: query_id incorrecto")
  docs=row.get("documents",[]); frags=row.get("fragments",[])
  if len(docs)!=3 or len(frags)!=10: errors.append(f"Linea {i}: se requieren 3 documentos y 10 fragments")
  if [x.get("rank") for x in docs] != [1,2,3]: errors.append(f"Linea {i}: ranks de documentos invalidos")
  if [x.get("rank") for x in frags] != list(range(1,11)): errors.append(f"Linea {i}: ranks de fragmentos invalidos")
  if len({x.get("doc_id") for x in docs}) != len(docs): errors.append(f"Linea {i}: documentos repetidos")
  for f in frags:
   if not {"rank","chunk_id","doc_id","text"}.issubset(f): errors.append(f"Linea {i}: fragmento incompleto")
   if known_chunk_ids is not None and f.get("chunk_id") not in known_chunk_ids: errors.append(f"Linea {i}: chunk inexistente")
   if known_doc_ids is not None and f.get("doc_id") not in known_doc_ids: errors.append(f"Linea {i}: documento de fragmento inexistente")
  if known_doc_ids is not None and any(x.get("doc_id") not in known_doc_ids for x in docs): errors.append(f"Linea {i}: documento inexistente")
  if any(word_count(x.get("text",""))>250 for x in frags): errors.append(f"Linea {i}: fragmento >250 palabras")
 return errors
