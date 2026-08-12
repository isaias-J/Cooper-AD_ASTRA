from __future__ import annotations
import json, logging, platform, time
from pathlib import Path
import numpy as np
from .cache import EmbeddingCache

DEFAULT_MODEL="intfloat/multilingual-e5-base"
LOGGER = logging.getLogger(__name__)

def select_device(requested: str | None, cuda_available: bool) -> str:
    """Resolve the inference device, failing loudly for an unavailable explicit CUDA request."""
    if requested not in {None, "auto", "cpu", "cuda"}:
        raise ValueError("device debe ser auto, cpu o cuda")
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("Se solicitó CUDA, pero torch.cuda.is_available() es False. Ejecute scripts/gpu_smoke_test.py.")
    if requested in {"cpu", "cuda"}:
        return requested
    return "cuda" if cuda_available else "cpu"

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("El encoder produjo al menos un embedding de norma cero")
    return np.ascontiguousarray(vectors / norms, dtype="float32")

class Encoder:
    def __init__(self, model=DEFAULT_MODEL, device="auto", use_fp16=True, cache_dir: Path | None = None):
        import torch
        from sentence_transformers import SentenceTransformer
        self.torch=torch
        self.requested_device=device
        self.device=select_device(device, torch.cuda.is_available())
        self.model_name=model; self.model=SentenceTransformer(model,device=self.device)
        self.use_fp16=bool(use_fp16 and self.device == "cuda")
        if self.use_fp16:
            self.model.half()
        self.tokenizer=self.model.tokenizer
        self.embedding_cache = EmbeddingCache(cache_dir) if cache_dir else None
    def encode(self, values, kind, batch_size=32):
        if kind not in {"query", "passage"}: raise ValueError("kind debe ser query o passage")
        if batch_size < 1: raise ValueError("batch_size debe ser positivo")
        prefix="query: " if kind=="query" else "passage: "
        prepared=[prefix+x for x in values]
        cached = {}
        hashes = []
        if self.embedding_cache:
            hashes, cached = self.embedding_cache.get_many(self.model_name, kind, prepared)
        misses = [value for value in prepared if not cached or self.embedding_cache._hash(value) not in cached]
        current=batch_size
        while True:
            try:
                encoded = self.model.encode(misses,batch_size=current,normalize_embeddings=False,
                    show_progress_bar=len(misses)>32,convert_to_numpy=True) if misses else np.empty((0, self.model.get_sentence_embedding_dimension()), dtype="float32")
                encoded = l2_normalize(encoded) if len(encoded) else encoded
                if self.embedding_cache and misses:
                    self.embedding_cache.put_many(self.model_name, kind, misses, encoded)
                    _, cached = self.embedding_cache.get_many(self.model_name, kind, prepared)
                arr = np.asarray([cached[content_hash] for content_hash in hashes], dtype="float32") if self.embedding_cache else encoded
                self.last_batch_size=current
                return l2_normalize(arr)
            except self.torch.cuda.OutOfMemoryError as exc:
                if self.device != "cuda" or current == 1: raise RuntimeError("CUDA OOM incluso con batch_size=1") from exc
                reduced=max(1,current//2)
                LOGGER.warning("CUDA OOM con batch_size=%d; reintentando con batch_size=%d",current,reduced)
                self.torch.cuda.empty_cache(); current=reduced
    def config(self):
        return {"model":self.model_name,"requested_device":self.requested_device,"build_device":self.device,
            "dtype":"float16" if self.use_fp16 else "float32","dimension":self.model.get_sentence_embedding_dimension(),
            "normalized":True,"similarity":"inner_product","index_type":"IndexFlatIP",
            "prefixes":{"passage":"passage: ","query":"query: "},"cuda_available":self.torch.cuda.is_available(),
            "gpu":self.torch.cuda.get_device_name(0) if self.torch.cuda.is_available() else None,"platform":platform.platform(),
            "embedding_cache":bool(self.embedding_cache)}

def build_index(chunks, out_dir:Path, encoder:Encoder, batch_size=32):
    import faiss
    out_dir.mkdir(parents=True,exist_ok=True); started=time.time(); metadata=list(chunks)
    vectors=encoder.encode([x["texto"] for x in metadata],"passage",batch_size)
    if len(vectors) != len(metadata): raise RuntimeError("FAISS y metadata quedarian desalineados: el encoder no devolvio un vector por chunk")
    index=faiss.IndexFlatIP(vectors.shape[1]); index.add(vectors); faiss.write_index(index,str(out_dir/"index.faiss"))
    with (out_dir/"metadata.jsonl").open("w",encoding="utf-8") as f:
        for item in metadata: f.write(json.dumps(item,ensure_ascii=False)+"\n")
    if index.ntotal != len(metadata): raise RuntimeError("FAISS y metadata quedaron desalineados durante la construcción")
    config=encoder.config()|{"chunks":len(metadata),"batch_size_requested":batch_size,
        "batch_size_effective":getattr(encoder,"last_batch_size",batch_size),"build_seconds":round(time.time()-started,2)}
    (out_dir/"encoder_config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    return config
