from __future__ import annotations
import argparse, collections, json
from pathlib import Path
from .core import SUPPORTED, normalized_rel, phenomenon, stable_doc_id, write_jsonl

def audit(root: Path, report_dir: Path, manifest_path: Path) -> dict:
    if not root.is_dir(): raise FileNotFoundError(f"CORPUS_ROOT no existe: {root}")
    records=[]; by_ext=collections.Counter(); by_f=collections.Counter(); errors=[]
    for p in sorted(x for x in root.rglob("*") if x.is_file() and x.name != ".DS_Store"):
        rel=normalized_rel(p, root); ext=p.suffix.lower() or "[sin_extension]"; size=p.stat().st_size
        f=phenomenon(rel); indexable=bool(f and ext in SUPPORTED and size); status="pendiente" if indexable else "excluido"
        note=[]
        if ext not in SUPPORTED: note.append("Formato no soportado por baseline")
        if not f: note.append("Archivo de control sin fenomeno: excluido del indice para cumplir metadata obligatoria")
        if not size: errors.append(rel); note.append("Archivo vacio")
        item={"ruta_relativa":rel,"extension":ext,"formato":ext.lstrip("."),"tamano_bytes":size,
              "fenomeno":f,"indexable":indexable,"estado_extraccion":status,"observaciones":"; ".join(note),"doc_id":stable_doc_id(rel)}
        records.append(item); by_ext[ext]+=1; by_f[str(f) if f else "pendiente"]+=1
    payload={"corpus_root":str(root),"total_documentos":len(records),"por_formato":dict(by_ext),"por_fenomeno":dict(by_f),"archivos_vacios":errors,"archivos":records}
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir/"corpus_inventory.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    rows=["# Inventario del corpus", "", f"Documentos: **{len(records)}**", "", "## Por formato", "", "| Formato | Cantidad |", "|---|---:|"]
    rows += [f"| {k} | {v} |" for k,v in sorted(by_ext.items())]
    rows += ["", "## Por fenomeno", "", "| Fenomeno | Cantidad |", "|---|---:|"] + [f"| {k} | {v} |" for k,v in sorted(by_f.items())]
    if errors: rows += ["", "## Archivos vacios", ""] + [f"- `{x}`" for x in errors]
    (report_dir/"corpus_inventory.md").write_text("\n".join(rows)+"\n",encoding="utf-8")
    write_jsonl(manifest_path, records)
    return payload

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--corpus-root",type=Path,required=True); p.add_argument("--report-dir",type=Path,default=Path("reports")); p.add_argument("--manifest",type=Path,default=Path("data/processed/documents_manifest.jsonl")); a=p.parse_args(); print(json.dumps(audit(a.corpus_root,a.report_dir,a.manifest),ensure_ascii=False,indent=2))
