#!/usr/bin/env python3
"""Copies final artifacts and the local package needed by generador.py."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("entrega"))
    parser.add_argument("--graph", type=Path, default=None)
    args = parser.parse_args()
    existing = [item for item in args.out.iterdir() if item.name != "ESTRUCTURA"] if args.out.exists() else []
    if existing:
        raise RuntimeError("entrega no esta vacia; limpiela manualmente para evitar mezcla de artefactos")
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.results, args.out / "resultados.jsonl")
    shutil.copy2(Path(__file__).parents[1] / "generador.py", args.out / "generador.py")
    shutil.copy2(args.report, args.out / "informe_tecnico.pdf")
    target = args.out / "base_vectorial" / args.index_dir.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.index_dir, target)
    if args.graph:
        graph_target = args.out / "base_vectorial" / "grafo"
        graph_target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.graph, graph_target / "grafo.graphml")
    shutil.copytree(Path(__file__).parents[1] / "src" / "codefest", args.out / "src" / "codefest")


if __name__ == "__main__":
    main()
