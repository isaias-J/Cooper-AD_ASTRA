#!/usr/bin/env python3
from __future__ import annotations
import argparse, platform, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from codefest.vector import DEFAULT_MODEL, Encoder

def main() -> None:
    parser=argparse.ArgumentParser(description="Prueba CUDA + encoder + FAISS sin indexar el corpus")
    parser.add_argument("--device",choices=["auto","cuda","cpu"],default="cuda")
    parser.add_argument("--model",default=DEFAULT_MODEL)
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--no-fp16",action="store_true")
    args=parser.parse_args()
    import faiss, torch
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA incluida en PyTorch: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("ERROR: se solicitó --device cuda pero CUDA no está activa. Revise nvidia-smi y la instalación del wheel CUDA de PyTorch.")
    if torch.cuda.is_available():
        free,total=torch.cuda.mem_get_info()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU libre/total: {free/2**30:.2f}/{total/2**30:.2f} GiB")
    encoder=Encoder(args.model,device=args.device,use_fp16=not args.no_fp16)
    print(f"Encoder cargado: {encoder.model_name}")
    print(f"Dimensión: {encoder.model.get_sentence_embedding_dimension()}")
    print(f"Dispositivo efectivo: {encoder.device}; dtype: {'float16' if encoder.use_fp16 else 'float32'}")
    phrases=["La seguridad aeroespacial requiere cooperación.","Space security requires cooperation.","A segurança espacial requer cooperação."]
    started=time.perf_counter(); vectors=encoder.encode(phrases,"passage",args.batch_size)
    if encoder.device == "cuda": torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    print(f"Embeddings ES/EN/PT: shape={vectors.shape}; tiempo={elapsed:.3f}s; batch efectivo={encoder.last_batch_size}")
    norms=np.linalg.norm(vectors,axis=1)
    if not np.allclose(norms,1.0,atol=1e-5): raise SystemExit(f"ERROR: embeddings no normalizados: {norms}")
    index=faiss.IndexFlatIP(vectors.shape[1]); index.add(vectors)
    query=encoder.encode(["cooperación para seguridad espacial"],"query",args.batch_size)
    scores,ids=index.search(query,3)
    if index.ntotal != 3 or ids.shape != (1,3): raise SystemExit("ERROR: búsqueda FAISS mínima falló")
    print(f"FAISS IndexFlatIP: OK; ntotal={index.ntotal}; ids={ids[0].tolist()}; scores={[round(float(x),4) for x in scores[0]]}")

if __name__ == "__main__": main()
