from __future__ import annotations

import csv
import json
from pathlib import Path

from .core import clean_text

TEXT_KEYS = ("title", "body", "body_text", "body_paragraphs", "content", "text", "description", "summary", "headline")


def _json_blocks(value, path="$"):
    """Yield independent text fields, preserving their JSON path instead of flattening a file."""
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key.lower() in TEXT_KEYS and isinstance(child, (str, int, float)):
                yield f"{key}: {child}", {"json_path": child_path}
            else:
                yield from _json_blocks(child, child_path)
    elif isinstance(value, list):
        for number, child in enumerate(value):
            yield from _json_blocks(child, f"{path}[{number}]")


def _pdf_blocks(path: Path):
    import fitz

    document = fitz.open(path)
    try:
        candidates = []
        raw_pages = []
        for page_number, page in enumerate(document, 1):
            page_blocks = []
            height = page.rect.height
            for block_number, block in enumerate(page.get_text("blocks", sort=True)):
                text = clean_text(block[4])
                if not text:
                    continue
                page_blocks.append((text, block_number))
                if len(text) <= 120 and (block[1] < 72 or block[3] > height - 72):
                    candidates.append(text)
            raw_pages.append(page_blocks)
        repeated = {text for text in candidates if candidates.count(text) >= max(3, len(raw_pages) // 3)}
        blocks = []
        for page_number, page_blocks in enumerate(raw_pages, 1):
            for text, block_number in page_blocks:
                if text not in repeated:
                    blocks.append((text, {"page_start": page_number, "page_end": page_number, "block_number": block_number}))
        return blocks, {"page_start": 1, "page_end": len(raw_pages), "removed_repeated_blocks": len(repeated)}
    finally:
        document.close()


def _pbf_blocks(path: Path):
    import mapbox_vector_tile

    tile = mapbox_vector_tile.decode(path.read_bytes())
    blocks, seen = [], set()
    for layer, payload in tile.items():
        for feature in payload.get("features", []):
            properties = feature.get("properties") or {}
            key = (layer, tuple(sorted((str(k), str(v)) for k, v in properties.items())))
            if key in seen or not properties:
                continue
            seen.add(key)
            text = "; ".join([f"layer: {layer}"] + [f"{key}: {value}" for key, value in properties.items()])
            blocks.append((text, {"layer": layer, "feature_id": feature.get("id"), "unit_type": "row"}))
    return blocks


def extract(path: Path, enable_ocr=False) -> dict:
    """Extract source units without joining pages, JSON fields, list items, or table rows."""
    ext = path.suffix.lower()
    metadata = {"source_name": path.name, "page_start": None, "page_end": None}
    blocks: list[tuple[str, dict]] = []
    if ext in {".txt", ".md"}:
        blocks = [(path.read_text(encoding="utf-8", errors="replace"), {})]
    elif ext == ".json":
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        blocks = list(_json_blocks(value))
        metadata["json_type"] = type(value).__name__
        if isinstance(value, dict):
            for key in ("url", "date", "published", "tags"):
                if key in value:
                    metadata[key] = value[key]
    elif ext == ".csv":
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), 2):
                blocks.append(("; ".join(f"{key}: {value}" for key, value in row.items() if value not in (None, "")), {"row_number": row_number, "unit_type": "row"}))
    elif ext == ".xlsx":
        import openpyxl

        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        for sheet in workbook.worksheets:
            rows = sheet.iter_rows(values_only=True)
            headers = next(rows, ())
            for row_number, row in enumerate(rows, 2):
                text = "; ".join(f"{headers[i]}: {value}" for i, value in enumerate(row) if i < len(headers) and headers[i] and value not in (None, ""))
                blocks.append((text, {"sheet": sheet.title, "row_number": row_number, "unit_type": "row"}))
    elif ext == ".pdf":
        blocks, pdf_metadata = _pdf_blocks(path)
        metadata.update(pdf_metadata)
    elif ext in {".html", ".htm"}:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
        for node in soup(["script", "style", "nav", "footer", "header"]):
            node.decompose()
        blocks = [(node.get_text(" ", strip=True), {"unit_type": "list_item" if node.name == "li" else "paragraph"}) for node in soup.find_all(["h1", "h2", "h3", "p", "li"])]
    elif ext in {".jpg", ".jpeg", ".png", ".avif"}:
        if not enable_ocr:
            raise RuntimeError("OCR omitido por defecto: active --enable-ocr solo tras verificar texto relevante")
        import pytesseract
        from PIL import Image

        blocks = [(pytesseract.image_to_string(Image.open(path)), {"ocr_engine": "tesseract"})]
    elif ext == ".pbf":
        blocks = _pbf_blocks(path)
        metadata["pbf_features"] = len(blocks)
    else:
        raise RuntimeError(f"Formato no soportado: {ext}")
    return {
        "blocks": [{"text": clean_text(text), "metadata": metadata | extra} for text, extra in blocks if clean_text(text)],
        "metadata": metadata,
    }
