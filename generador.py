#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/"src"))
from codefest.vector import Encoder
from codefest.retrieval import Retriever
from codefest.chunking import sentences, word_count
from codefest.validate import validate_results

def output_texts(text):
 """Split the presentation text at sentence boundaries; never truncate a sentence."""
 kept=[]; out=[]
 for s in sentences(text):
  if word_count(s)>250:
   raise ValueError("Oracion individual >250 palabras: no puede emitirse sin incumplir completitud")
  if kept and word_count(" ".join(kept+[s]))>250:
   out.append(" ".join(kept)); kept=[]
  kept.append(s)
 if kept: out.append(" ".join(kept))
 return out
def main():
 p=argparse.ArgumentParser(); p.add_argument("--queries",type=Path,required=True); p.add_argument("--index-dir",type=Path,required=True); p.add_argument("--output",type=Path,default=Path("resultados.jsonl")); p.add_argument("--aggregation",choices=["max","top2sum","mean"],default="max"); p.add_argument("--model",default="intfloat/multilingual-e5-base"); a=p.parse_args()
 queries=[json.loads(x) for x in a.queries.read_text(encoding="utf-8").splitlines() if x.strip()]
 r=Retriever(a.index_dir,Encoder(a.model)); rows=[]
 for q in queries:
  query=q.get("query") or q.get("text") or q.get("consulta")
  if not query: raise ValueError(f"Consulta sin texto: {q}")
  docs,hits=r.search(query,aggregation=a.aggregation)
  emitted=[]
  for hit in hits:
   for text in output_texts(hit["texto"]):
    emitted.append({"chunk_id":hit["chunk_id"],"doc_id":hit["doc_id"],"text":text})
    if len(emitted)==10: break
   if len(emitted)==10: break
  if len(docs)<3 or len(emitted)<10: raise ValueError("Indice insuficiente para devolver 3 documentos y 10 fragmentos validos")
  rows.append({"query_id":q["query_id"],"documents":[{"rank":i,"doc_id":d} for i,d in enumerate(docs,1)],"fragments":[x|{"rank":i} for i,x in enumerate(emitted,1)]})
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
 errors=validate_results(a.output,official=len(queries)==50,known_chunk_ids={x["chunk_id"] for x in r.metadata},known_doc_ids={x["doc_id"] for x in r.metadata})
 if errors: raise ValueError("; ".join(errors))
 print(f"Escrito y validado: {a.output} ({len(rows)} consultas)")
if __name__=="__main__": main()
