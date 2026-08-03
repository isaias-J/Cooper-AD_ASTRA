#!/usr/bin/env python3
"""Copies only final required artifacts into entrega/ after a successful preflight."""
from __future__ import annotations
import argparse, shutil
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,required=True); p.add_argument("--index-dir",type=Path,required=True); p.add_argument("--report",type=Path,required=True); p.add_argument("--out",type=Path,default=Path("entrega")); a=p.parse_args()
 if a.out.exists() and any(a.out.iterdir()): raise RuntimeError("entrega no esta vacia; limpiela manualmente para evitar mezcla de artefactos")
 a.out.mkdir(parents=True,exist_ok=True); shutil.copy2(a.results,a.out/"resultados.jsonl"); shutil.copy2("generador.py",a.out/"generador.py"); shutil.copy2(a.report,a.out/"informe_tecnico.pdf")
 target=a.out/"base_vectorial"/a.index_dir.name; target.parent.mkdir(); shutil.copytree(a.index_dir,target)
if __name__=="__main__": main()
