from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

class Retriever:
 def __init__(self,index_dir,encoder):
  import faiss
  self.index=faiss.read_index(str(Path(index_dir)/"index.faiss")); self.encoder=encoder
  self.metadata=[json.loads(x) for x in (Path(index_dir)/"metadata.jsonl").read_text(encoding="utf-8").splitlines() if x]
  if self.index.ntotal!=len(self.metadata): raise ValueError("FAISS y metadata no estan alineados")
 def search(self,query,top_chunks=10,candidates=100,aggregation="max"):
  scores,ids=self.index.search(self.encoder.encode([query],"query"),min(candidates,self.index.ntotal)); hits=[]
  for score,i in zip(scores[0],ids[0]):
   if i>=0: hits.append((float(score),self.metadata[int(i)]))
  grouped=defaultdict(list)
  for score,m in hits: grouped[m["doc_id"]].append(score)
  def agg(xs): return max(xs) if aggregation=="max" else sum(sorted(xs,reverse=True)[:2]) if aggregation=="top2sum" else sum(xs)/len(xs)
  docs=[d for d,_ in sorted(grouped.items(),key=lambda kv:agg(kv[1]),reverse=True)[:3]]
  return docs,[m|{"score":s} for s,m in hits[:top_chunks]]
