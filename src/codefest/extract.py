from __future__ import annotations

import csv
import json
import os
import shutil
import re
from collections import defaultdict
from pathlib import Path

from .core import clean_text

TEXT_KEYS = ("title", "body", "body_text", "body_paragraphs", "content", "text", "description", "summary", "headline")


def remove_repeated_ocr_blocks(extracted: dict) -> dict:
    """Remove short OCR boilerplate repeated across many pages of one document."""
    blocks = extracted.get("blocks", [])
    if not blocks or not any(block.get("metadata", {}).get("ocr_engine") for block in blocks):
        return extracted
    page_count = max((block["metadata"].get("page_end") or 1 for block in blocks), default=1)
    threshold = max(3, page_count // 3)

    def canonical(text: str) -> str:
        value = re.sub(r"\s+", " ", text.lower()).strip(" .,-:;")
        if re.search(r"\b(?:page|página|pagina)\s+\d+\s+(?:of|de)\s+\d+\b", value):
            value = re.sub(r"\d+", "#", value)
        return value

    counts: defaultdict[str, int] = defaultdict(int)
    for block in blocks:
        if len(block["text"]) <= 250:
            counts[canonical(block["text"])] += 1
    repeated = {value for value, count in counts.items() if len(value) >= 8 and count >= threshold}
    kept = [block for block in blocks if canonical(block["text"]) not in repeated]
    removed = len(blocks) - len(kept)
    if not removed:
        return extracted
    output = dict(extracted)
    output["metadata"] = dict(extracted.get("metadata", {})) | {"removed_repeated_ocr_blocks": removed}
    output["blocks"] = [
        dict(block) | {"metadata": dict(block["metadata"]) | {"removed_repeated_ocr_blocks": removed}}
        for block in kept
    ]
    return output


def _json_blocks(value, path="$"):
    """Yield text fields, including strings nested in paragraph/list arrays."""
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key.lower() in TEXT_KEYS and isinstance(child, (str, int, float)):
                yield f"{key}: {child}", {"json_path": child_path}
            elif key.lower() in TEXT_KEYS and isinstance(child, list):
                for item_number, item in enumerate(child):
                    if isinstance(item, (str, int, float)):
                        yield f"{key}: {item}", {"json_path": f"{child_path}[{item_number}]"}
                    else:
                        yield from _json_blocks(item, f"{child_path}[{item_number}]")
            else:
                yield from _json_blocks(child, child_path)
    elif isinstance(value, list):
        for number, child in enumerate(value):
            yield from _json_blocks(child, f"{path}[{number}]")


def _resolve_tesseract(command: Path | None = None) -> str:
    candidates = [
        str(command) if command else None,
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    selected = next((item for item in candidates if item and Path(item).is_file()), None)
    if not selected:
        raise RuntimeError("Tesseract no encontrado. Instálelo o use --tesseract-command.")
    return selected


def _ocr_image_blocks(image, *, languages: str, command: Path | None, tessdata_dir: Path | None,
                      psm: int, min_confidence: float):
    import pytesseract
    from pytesseract import Output

    pytesseract.pytesseract.tesseract_cmd = _resolve_tesseract(command)
    config = f"--oem 1 --psm {psm}"
    previous_tessdata = os.environ.get("TESSDATA_PREFIX")
    if tessdata_dir:
        os.environ["TESSDATA_PREFIX"] = str(tessdata_dir.resolve())
    try:
        data = pytesseract.image_to_data(image, lang=languages, config=config, output_type=Output.DICT)
    except pytesseract.TesseractError as exc:
        raise RuntimeError(f"OCR falló para idiomas {languages}: {exc}") from exc
    finally:
        if tessdata_dir:
            if previous_tessdata is None:
                os.environ.pop("TESSDATA_PREFIX", None)
            else:
                os.environ["TESSDATA_PREFIX"] = previous_tessdata
    lines: defaultdict[tuple[int, int, int], list[tuple[str, float]]] = defaultdict(list)
    for number, raw_text in enumerate(data["text"]):
        text = clean_text(raw_text)
        if not text:
            continue
        try:
            confidence = float(data["conf"][number])
        except (TypeError, ValueError):
            confidence = -1.0
        key = (int(data["block_num"][number]), int(data["par_num"][number]), int(data["line_num"][number]))
        lines[key].append((text, confidence))
    paragraphs: defaultdict[tuple[int, int], list[tuple[int, str, list[float]]]] = defaultdict(list)
    for (block, paragraph, line), words in lines.items():
        paragraphs[(block, paragraph)].append((line, " ".join(word for word, _ in words), [score for _, score in words if score >= 0]))
    output = []
    for (block, paragraph), values in sorted(paragraphs.items()):
        values.sort(key=lambda item: item[0])
        text = clean_text(" ".join(item[1] for item in values))
        confidence_values = [score for item in values for score in item[2]]
        mean_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else None
        if text and (mean_confidence is None or mean_confidence >= min_confidence):
            output.append((text, {
                "ocr_engine": "tesseract-5",
                "ocr_languages": languages,
                "ocr_confidence": round(mean_confidence, 2) if mean_confidence is not None else None,
                "ocr_block": block,
                "ocr_paragraph": paragraph,
                "unit_type": "ocr_paragraph",
            }))
    return output


def _pdf_blocks(path: Path, *, enable_ocr=False, ocr_languages="spa+eng+por", ocr_dpi=250,
                tesseract_command: Path | None = None, tessdata_dir: Path | None = None,
                ocr_min_confidence=60.0):
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
        if blocks or not enable_ocr:
            return blocks, {"page_start": 1, "page_end": len(raw_pages), "removed_repeated_blocks": len(repeated), "ocr_applied": False}
        from PIL import Image

        ocr_blocks = []
        scale = ocr_dpi / 72
        for page_number, page in enumerate(document, 1):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False, colorspace=fitz.csRGB)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            for text, extra in _ocr_image_blocks(
                image, languages=ocr_languages, command=tesseract_command, tessdata_dir=tessdata_dir,
                psm=3, min_confidence=ocr_min_confidence,
            ):
                ocr_blocks.append((text, extra | {"page_start": page_number, "page_end": page_number}))
        return ocr_blocks, {
            "page_start": 1,
            "page_end": len(raw_pages),
            "removed_repeated_blocks": 0,
            "ocr_applied": True,
            "ocr_dpi": ocr_dpi,
        }
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


def extract(path: Path, enable_ocr=False, *, ocr_languages="spa+eng+por", ocr_dpi=250,
            tesseract_command: Path | None = None, tessdata_dir: Path | None = None,
            ocr_min_confidence=60.0) -> dict:
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
        blocks, pdf_metadata = _pdf_blocks(
            path,
            enable_ocr=enable_ocr,
            ocr_languages=ocr_languages,
            ocr_dpi=ocr_dpi,
            tesseract_command=tesseract_command,
            tessdata_dir=tessdata_dir,
            ocr_min_confidence=ocr_min_confidence,
        )
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
        from PIL import Image

        with Image.open(path) as image:
            blocks = _ocr_image_blocks(
                image.convert("RGB"), languages=ocr_languages, command=tesseract_command,
                tessdata_dir=tessdata_dir, psm=6, min_confidence=ocr_min_confidence,
            )
        metadata.update({"ocr_applied": True, "ocr_dpi": None})
    elif ext == ".pbf":
        blocks = _pbf_blocks(path)
        metadata["pbf_features"] = len(blocks)
    else:
        raise RuntimeError(f"Formato no soportado: {ext}")
    return {
        "blocks": [{"text": clean_text(text), "metadata": metadata | extra} for text, extra in blocks if clean_text(text)],
        "metadata": metadata,
    }
