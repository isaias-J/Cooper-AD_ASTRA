from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

class Retriever:
    def __init__(self, index_dir, encoder, graph_path=None):
        import faiss
        import networkx as nx

        path = Path(index_dir)
        self.index = faiss.read_index(str(path / "index.faiss"))
        self.encoder = encoder
        self.graph = nx.read_graphml(graph_path) if graph_path else None
        with (path / "metadata.jsonl").open(encoding="utf-8") as stream:
            self.metadata = [json.loads(line) for line in stream if line.strip()]
        self.by_chunk = {item["chunk_id"]: item for item in self.metadata}
        self.graph_evidence_index = None
        if self.graph is not None:
            from .graph import build_graph_evidence_index
            self.graph_evidence_index = build_graph_evidence_index(self.graph)
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

    def search(self, query, top_chunks=10, candidates=100, aggregation="max", batch_size=32, graph_weight=0.0025):
        if graph_weight < 0: raise ValueError("graph_weight no puede ser negativo")
        scores, ids = self.index.search(
            self.encoder.encode([query], "query", batch_size=batch_size),
            min(candidates, self.index.ntotal),
        )
        hits = []
        for score, i in zip(scores[0], ids[0]):
            if i >= 0:
                hits.append((float(score), self.metadata[int(i)]))
        if self.graph is not None:
            from .graph import indexed_graph_chunk_scores

            graph_scores = indexed_graph_chunk_scores(self.graph_evidence_index, query)
            existing = {metadata["chunk_id"] for _, metadata in hits}
            for chunk_id, graph_score in graph_scores.items():
                if chunk_id not in existing and chunk_id in self.by_chunk:
                    # Start at zero; the common fusion pass applies the graph contribution once.
                    hits.append((0.0, self.by_chunk[chunk_id]))
            hits = [
                (score + graph_weight * min(graph_scores.get(metadata["chunk_id"], 0), 10), metadata)
                for score, metadata in hits
            ]
            hits.sort(key=lambda item: item[0], reverse=True)
        grouped = defaultdict(list)
        for score, metadata in hits:
            grouped[metadata["doc_id"]].append(score)
        def agg(values):
            return max(values) if aggregation == "max" else sum(sorted(values, reverse=True)[:2]) if aggregation == "top2sum" else sum(values) / len(values)
        docs = [doc_id for doc_id, _ in sorted(grouped.items(), key=lambda item: agg(item[1]), reverse=True)[:3]]
        return docs, [metadata | {"score": score} for score, metadata in hits[:top_chunks]]
