from __future__ import annotations
import csv, json, re
from pathlib import Path
from .core import clean_text

TEXT_KEYS=("title","body","body_text","body_paragraphs","content","text","description","summary","headline")

def _flatten_json(value, key=""):
    if isinstance(value,dict):
        for k,v in value.items(): yield from _flatten_json(v,k)
    elif isinstance(value,list):
        for v in value: yield from _flatten_json(v,key)
    elif isinstance(value,(str,int,float)) and key.lower() in TEXT_KEYS: yield f"{key}: {value}"

def extract(path: Path) -> dict:
    """Return traceable original text and lightweight source metadata; never synthesizes content."""
    ext=path.suffix.lower(); meta={"source_name":path.name,"page_start":None,"page_end":None}
    if ext in {".txt",".md"}: text=path.read_text(encoding="utf-8",errors="replace")
    elif ext==".json":
        obj=json.loads(path.read_text(encoding="utf-8",errors="replace")); text="\n".join(_flatten_json(obj)); meta["json_type"]=type(obj).__name__
        for k in ("url","date","published","tags"): 
            if isinstance(obj,dict) and k in obj: meta[k]=obj[k]
    elif ext==".csv":
        with path.open(encoding="utf-8-sig",errors="replace",newline="") as f:
            rows=csv.DictReader(f); text="\n".join("; ".join(f"{k}: {v}" for k,v in row.items() if v not in (None,"")) for row in rows)
    elif ext==".xlsx":
        import openpyxl
        wb=openpyxl.load_workbook(path,read_only=True,data_only=True); parts=[]
        for ws in wb.worksheets:
            rows=ws.iter_rows(values_only=True); headers=next(rows,())
            for row in rows:
                parts.append("; ".join(f"{headers[i]}: {v}" for i,v in enumerate(row) if i<len(headers) and headers[i] and v not in (None,"")))
        text="\n".join(parts)
    elif ext==".pdf":
        import fitz
        doc=fitz.open(path); pages=[page.get_text("text",sort=True) for page in doc]; text="\n\n".join(pages); meta.update(page_start=1,page_end=len(pages))
    elif ext in {".html",".htm"}:
        from bs4 import BeautifulSoup
        soup=BeautifulSoup(path.read_text(encoding="utf-8",errors="replace"),"html.parser")
        for node in soup(["script","style","nav","footer","header"]): node.decompose()
        text="\n".join(x.get_text(" ",strip=True) for x in soup.find_all(["h1","h2","h3","p","li"]))
    elif ext in {".jpg",".jpeg",".png",".avif"}: raise RuntimeError("OCR no habilitado: instale/configure Tesseract y habilitelo explicitamente")
    elif ext==".pbf": raise RuntimeError("PBF detectado: requiere especificacion del esquema/datos geoespaciales; no se inventa extraccion")
    else: raise RuntimeError(f"Formato no soportado: {ext}")
    return {"text":clean_text(text),"metadata":meta}
