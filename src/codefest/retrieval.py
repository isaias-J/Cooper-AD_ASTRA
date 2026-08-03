from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

class Retriever:
    def __init__(self, index_dir, encoder):
        import faiss

        path = Path(index_dir)
        self.index = faiss.read_index(str(path / "index.faiss"))
        self.encoder = encoder
        with (path / "metadata.jsonl").open(encoding="utf-8") as stream:
            self.metadata = [json.loads(line) for line in stream if line.strip()]
        if self.index.ntotal != len(self.metadata):
            raise ValueError("FAISS y metadata no estan alineados")
        config_path = path / "encoder_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("model") != encoder.model_name:
                raise ValueError(f"Encoder incompatible: índice={config.get('model')}, consulta={encoder.model_name}")
            if config.get("dimension") != self.index.d:
                raise ValueError("Dimensión declarada no coincide con FAISS")
            if config.get("prefixes") != {"passage": "passage: ", "query": "query: "}:
                raise ValueError("Prefijos del índice incompatibles")

    def search(self, query, top_chunks=10, candidates=100, aggregation="max"):
        scores, ids = self.index.search(self.encoder.encode([query], "query"), min(candidates, self.index.ntotal))
        hits = []
        for score, i in zip(scores[0], ids[0]):
            if i >= 0:
                hits.append((float(score), self.metadata[int(i)]))
        grouped = defaultdict(list)
        for score, metadata in hits:
            grouped[metadata["doc_id"]].append(score)
        def agg(values):
            return max(values) if aggregation == "max" else sum(sorted(values, reverse=True)[:2]) if aggregation == "top2sum" else sum(values) / len(values)
        docs = [doc_id for doc_id, _ in sorted(grouped.items(), key=lambda item: agg(item[1]), reverse=True)[:3]]
        return docs, [metadata | {"score": score} for score, metadata in hits[:top_chunks]]
