from __future__ import annotations
import json, platform, time
from pathlib import Path
import numpy as np

DEFAULT_MODEL="intfloat/multilingual-e5-base"
class Encoder:
    def __init__(self, model=DEFAULT_MODEL, device=None):
        import torch
        from sentence_transformers import SentenceTransformer
        self.torch=torch; self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name=model; self.model=SentenceTransformer(model,device=self.device)
        self.tokenizer=self.model.tokenizer
    def encode(self, values, kind, batch_size=32):
        prefix="query: " if kind=="query" else "passage: "
        arr=self.model.encode([prefix+x for x in values],batch_size=batch_size,normalize_embeddings=True,show_progress_bar=True,convert_to_numpy=True)
        return np.asarray(arr,dtype="float32")
    def config(self):
        return {"model":self.model_name,"device":self.device,"dimension":self.model.get_sentence_embedding_dimension(),"normalized":True,"prefixes":{"passage":"passage: ","query":"query: "},"candidates":["intfloat/multilingual-e5-large","BAAI/bge-m3"],"cuda_available":self.torch.cuda.is_available(),"gpu":self.torch.cuda.get_device_name(0) if self.torch.cuda.is_available() else None,"platform":platform.platform()}

def build_index(chunks, out_dir:Path, encoder:Encoder, batch_size=32):
    import faiss
    out_dir.mkdir(parents=True,exist_ok=True); started=time.time(); metadata=list(chunks)
    vectors=encoder.encode([x["texto"] for x in metadata],"passage",batch_size)
    index=faiss.IndexFlatIP(vectors.shape[1]); index.add(vectors); faiss.write_index(index,str(out_dir/"index.faiss"))
    with (out_dir/"metadata.jsonl").open("w",encoding="utf-8") as f:
        for item in metadata: f.write(json.dumps(item,ensure_ascii=False)+"\n")
    config=encoder.config()|{"chunks":len(metadata),"build_seconds":round(time.time()-started,2)}
    (out_dir/"encoder_config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    return config
