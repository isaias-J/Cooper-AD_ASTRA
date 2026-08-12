#!/usr/bin/env python3
"""Build the optional GraphML knowledge graph from indexed metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from codefest.graph import write_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("base_vectorial/grafo/grafo.graphml"))
    parser.add_argument("--min-mentions", type=int, default=1)
    args = parser.parse_args()
    if args.min_mentions < 1:
        raise ValueError("--min-mentions debe ser positivo")
    print(json.dumps(write_graph(args.metadata, args.output, args.min_mentions), ensure_ascii=False))


if __name__ == "__main__":
    main()
