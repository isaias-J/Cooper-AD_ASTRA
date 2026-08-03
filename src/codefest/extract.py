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

def _pdf_text(path: Path):
    import fitz
    doc=fitz.open(path)
    try: raw=[page.get_text("text",sort=True).splitlines() for page in doc]
    finally: doc.close()
    # Running headers/footers are usually a page's first/last short line and recur.
    candidates=[line.strip() for page in raw for line in (page[:2]+page[-2:]) if line.strip()]
    repeated={x for x in candidates if candidates.count(x)>=max(3,len(raw)//3)}
    pages=["\n".join(line for line in page if line.strip() not in repeated) for page in raw]
    return "\n\n".join(pages), {"page_start":1,"page_end":len(pages),"removed_repeated_lines":len(repeated)}

def _pbf_blocks(path: Path):
    import mapbox_vector_tile
    tile=mapbox_vector_tile.decode(path.read_bytes()); blocks=[]; seen=set()
    for layer, payload in tile.items():
        for feature in payload.get("features",[]):
            properties=feature.get("properties") or {}
            key=(layer,tuple(sorted((str(k),str(v)) for k,v in properties.items())))
            if key in seen or not properties: continue
            seen.add(key); blocks.append(("; ".join([f"layer: {layer}"]+[f"{k}: {v}" for k,v in properties.items()]),{"layer":layer,"feature_id":feature.get("id")}))
    return blocks

def extract(path: Path, enable_ocr=False) -> dict:
    """Return traceable original text and lightweight source metadata; never synthesizes content."""
    ext=path.suffix.lower(); meta={"source_name":path.name,"page_start":None,"page_end":None}; blocks=[]
    if ext in {".txt",".md"}: text=path.read_text(encoding="utf-8",errors="replace")
    elif ext==".json":
        obj=json.loads(path.read_text(encoding="utf-8",errors="replace")); text="\n".join(_flatten_json(obj)); meta["json_type"]=type(obj).__name__
        for k in ("url","date","published","tags"): 
            if isinstance(obj,dict) and k in obj: meta[k]=obj[k]
    elif ext==".csv":
        with path.open(encoding="utf-8-sig",errors="replace",newline="") as f:
            rows=csv.DictReader(f); blocks=[("; ".join(f"{k}: {v}" for k,v in row.items() if v not in (None,"")),{"row_number":n}) for n,row in enumerate(rows,2)]
    elif ext==".xlsx":
        import openpyxl
        wb=openpyxl.load_workbook(path,read_only=True,data_only=True)
        for ws in wb.worksheets:
            rows=ws.iter_rows(values_only=True); headers=next(rows,())
            for row_num,row in enumerate(rows,2):
                blocks.append(("; ".join(f"{headers[i]}: {v}" for i,v in enumerate(row) if i<len(headers) and headers[i] and v not in (None,"")),{"sheet":ws.title,"row_number":row_num}))
    elif ext==".pdf":
        text,pdf_meta=_pdf_text(path); meta.update(pdf_meta)
    elif ext in {".html",".htm"}:
        from bs4 import BeautifulSoup
        soup=BeautifulSoup(path.read_text(encoding="utf-8",errors="replace"),"html.parser")
        for node in soup(["script","style","nav","footer","header"]): node.decompose()
        text="\n".join(x.get_text(" ",strip=True) for x in soup.find_all(["h1","h2","h3","p","li"]))
    elif ext in {".jpg",".jpeg",".png",".avif"}:
        if not enable_ocr: raise RuntimeError("OCR omitido por defecto: active --enable-ocr solo tras verificar texto relevante")
        import pytesseract
        from PIL import Image
        text=pytesseract.image_to_string(Image.open(path)); meta["ocr_engine"]="tesseract"
    elif ext==".pbf": blocks=_pbf_blocks(path); meta["pbf_features"] = len(blocks)
    else: raise RuntimeError(f"Formato no soportado: {ext}")
    if blocks:
        return {"blocks":[{"text":clean_text(text),"metadata":meta|extra} for text,extra in blocks if clean_text(text)],"metadata":meta}
    return {"blocks":[{"text":clean_text(text),"metadata":meta}],"metadata":meta}
