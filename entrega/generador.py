#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from codefest.chunking import sentences, word_count
from codefest.retrieval import Retriever
from codefest.validate import validate_results
from codefest.vector import Encoder


def output_texts(text):
    """Split presentation text only at sentence boundaries."""
    kept = []
    output = []
    for sentence in sentences(text):
        if word_count(sentence) > 250:
            raise ValueError("Oracion individual >250 palabras")
        if kept and word_count(" ".join(kept + [sentence])) > 250:
            output.append(" ".join(kept))
            kept = []
        kept.append(sentence)
    if kept:
        output.append(" ".join(kept))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("resultados.jsonl"))
    parser.add_argument("--aggregation", choices=["max", "top2sum", "mean"], default="max")
    parser.add_argument("--model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--candidates", type=int, default=1000)
    args = parser.parse_args()

    with args.queries.open(encoding="utf-8") as stream:
        queries = [json.loads(line) for line in stream if line.strip()]
    retriever = Retriever(args.index_dir, Encoder(args.model, device=args.device, use_fp16=not args.no_fp16))
    rows = []
    for query_record in queries:
        query = query_record.get("query") or query_record.get("text") or query_record.get("consulta")
        if not query:
            raise ValueError(f"Consulta sin texto: {query_record}")
        docs, hits = retriever.search(
            query,
            top_chunks=args.candidates,
            candidates=args.candidates,
            aggregation=args.aggregation,
            batch_size=args.batch_size,
        )
        emitted = []
        for hit in hits:
            try:
                presentation = output_texts(hit["texto"])
            except ValueError:
                continue
            for text in presentation:
                emitted.append({"chunk_id": hit["chunk_id"], "doc_id": hit["doc_id"], "text": text})
                if len(emitted) == 10:
                    break
            if len(emitted) == 10:
                break
        if len(docs) < 3 or len(emitted) < 10:
            raise ValueError("Indice insuficiente para devolver 3 documentos y 10 fragmentos validos")
        rows.append({
            "query_id": query_record["query_id"],
            "documents": [{"rank": rank, "doc_id": doc_id} for rank, doc_id in enumerate(docs, 1)],
            "fragments": [fragment | {"rank": rank} for rank, fragment in enumerate(emitted, 1)],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    errors = validate_results(
        args.output,
        official=len(queries) == 50,
        known_chunk_ids={item["chunk_id"] for item in retriever.metadata},
        known_doc_ids={item["doc_id"] for item in retriever.metadata},
    )
    if errors:
        raise ValueError("; ".join(errors))
    print(f"Escrito y validado: {args.output} ({len(rows)} consultas)")


if __name__ == "__main__":
    main()
