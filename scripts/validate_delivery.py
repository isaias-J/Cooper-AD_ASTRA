#!/usr/bin/env python3
"""Strict preflight validation for the required CODEFEST delivery layout."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from codefest.validate import validate_metadata, validate_results

def main():
 p=argparse.ArgumentParser(); p.add_argument("--delivery-dir",type=Path,default=Path("entrega")); a=p.parse_args(); root=a.delivery_dir
 required=[root/"resultados.jsonl",root/"generador.py",root/"informe_tecnico.pdf"]
 index_dirs=list((root/"base_vectorial").glob("encoder_*")) if (root/"base_vectorial").is_dir() else []
 errors=[f"Falta: {x}" for x in required if not x.exists()]
 if not index_dirs: errors.append("Falta al menos un base_vectorial/encoder_*")
 known_chunk_ids=set(); known_doc_ids=set()
 for d in index_dirs:
  index_path=d/"index.faiss"; meta=d/"metadata.jsonl"
  if not index_path.exists() or not meta.exists(): errors.append(f"Indice incompleto: {d}"); continue
  errors += [f"{d}: {x}" for x in validate_metadata(meta)]
  with meta.open(encoding="utf-8") as stream:
   for line in stream:
    if line.strip():
     record=json.loads(line); known_chunk_ids.add(record["chunk_id"]); known_doc_ids.add(record["doc_id"])
  import faiss
  index=faiss.read_index(str(index_path));
  with meta.open(encoding="utf-8") as stream: count=sum(1 for line in stream if line.strip())
  if index.ntotal != count: errors.append(f"{d}: ntotal={index.ntotal}, metadata={count}")
  if (root/"resultados.jsonl").exists(): errors += validate_results(root/"resultados.jsonl",official=True,known_chunk_ids=known_chunk_ids,known_doc_ids=known_doc_ids)
  graph_path=root/"base_vectorial"/"grafo"/"grafo.graphml"
  if graph_path.exists():
   import networkx as nx
   graph=nx.read_graphml(graph_path)
   if graph.number_of_nodes()==0 or graph.number_of_edges()==0: errors.append("Grafo GraphML vacio")
   for _,_,attrs in graph.edges(data=True):
    if not attrs.get("relation") or not attrs.get("chunk_ids") or not attrs.get("doc_ids"): errors.append("Grafo sin trazabilidad completa"); break
 if (root/"informe_tecnico.pdf").exists():
  import fitz
  if len(fitz.open(root/"informe_tecnico.pdf")) > 8: errors.append("informe_tecnico.pdf excede 8 paginas")
 if errors:
  print("PRECHECK FAILED\n"+"\n".join(f"- {x}" for x in errors)); raise SystemExit(1)
 print("PRECHECK PASSED")
if __name__=="__main__": main()
