#!/usr/bin/env python3
"""Verify traceable Tesseract extraction on one PDF or image before a corpus build."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from codefest.extract import extract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--languages", default="spa+eng+por")
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--min-confidence", type=float, default=60.0)
    parser.add_argument("--tesseract-command", type=Path, default=None)
    parser.add_argument("--tessdata-dir", type=Path, default=None)
    args = parser.parse_args()
    result = extract(
        args.input,
        enable_ocr=True,
        ocr_languages=args.languages,
        ocr_dpi=args.dpi,
        ocr_min_confidence=args.min_confidence,
        tesseract_command=args.tesseract_command,
        tessdata_dir=args.tessdata_dir,
    )
    blocks = result["blocks"]
    if not blocks:
        raise RuntimeError("OCR no produjo bloques por encima del umbral de confianza")
    confidences = [block["metadata"].get("ocr_confidence") for block in blocks]
    confidences = [value for value in confidences if value is not None]
    payload = {
        "input": str(args.input.resolve()),
        "blocks": len(blocks),
        "characters": sum(len(block["text"]) for block in blocks),
        "pages": sorted({block["metadata"].get("page_start") for block in blocks if block["metadata"].get("page_start")}),
        "minimum_confidence": min(confidences) if confidences else None,
        "preview": " ".join(block["text"] for block in blocks)[:500],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
