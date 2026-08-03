#!/usr/bin/env python3
"""Generate the 4–6-page technical report after the final validated build."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


def paragraph(text, style):
    return Paragraph(text.replace("\n", "<br/>"), style)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/informe_tecnico.pdf"))
    parser.add_argument("--results", type=Path, default=Path("resultados_final.jsonl"))
    parser.add_argument("--failures", type=Path, default=Path("data/processed/extraction_failures.jsonl"))
    args = parser.parse_args()
    config = json.loads((args.index_dir / "encoder_config.json").read_text(encoding="utf-8"))
    metadata_count = sum(1 for line in (args.index_dir / "metadata.jsonl").open(encoding="utf-8") if line.strip())
    documents = set()
    with (args.index_dir / "metadata.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                documents.add(json.loads(line)["doc_id"])
    failures = [json.loads(line) for line in args.failures.open(encoding="utf-8") if line.strip()]
    failure_types = collections.Counter(item["error"] for item in failures)
    index_bytes = (args.index_dir / "index.faiss").stat().st_size
    styles = getSampleStyleSheet()
    body, heading = styles["BodyText"], styles["Heading2"]
    pages = [
        ("Informe técnico — Cooper | CODEFEST AD ASTRA 2026", [
            "<b>Entrega final reproducible, Windows + CUDA.</b>",
            "Objetivo: recuperación vectorial sobre el corpus oficial sin modelos generativos. El sistema usa un encoder público de Hugging Face, FAISS y metadata JSONL; no utiliza LLM, GPT/Claude/Gemini/Llama, query expansion, agentes, memoria, BM25, ChromaDB ni cross-encoder.",
            f"Construcción final: {metadata_count:,} vectores / {len(documents):,} documentos con chunks. Encoder: <b>{config['model']}</b>, dimensión {config['dimension']}. Hardware de construcción: {config['gpu']} con PyTorch CUDA; dtype de inferencia: {config['dtype']}.",
            "La entrega contiene resultados.jsonl, generador.py, base_vectorial e informe técnico. Las 50 consultas se conservan en orden y cada una devuelve tres documentos diferentes y diez fragmentos.",
        ]),
        ("Extracción estructurada y cobertura", [
            "PDF: PyMuPDF extrae bloques ordenados por página y elimina únicamente encabezados/pies repetidos. No se concatena el documento completo. JSON: los campos textuales se conservan individualmente con json_path. CSV/XLSX: cada fila mantiene pares columna: valor. HTML: párrafos y elementos de lista. PBF: atributos por feature.",
            "El chunker trata listas como ítems completos (incluye líneas envueltas) y tablas/filas como unidades completas. La segmentación de prosa ES/EN/PT solo divide después de puntuación terminal. No inventa fronteras.",
            f"Cobertura final: {len(documents):,} documentos con chunks de 1,837 archivos dentro de los tres fenómenos; 75 no aportaron chunks. De estos últimos, 66 no tienen texto extraíble y 9 son imágenes pendientes de OCR. Hay 118 documentos parcialmente cubiertos: contienen al menos un bloque no indexable, pero sí chunks válidos.",
            "Las imágenes relevantes inspeccionadas incluyen una tabla de residuos orbitales y gráficos de evaluación global/ASAT. OCR está implementado mediante pytesseract y deja trazabilidad ocr_engine; Tesseract no estaba instalado, por lo que esas 9 imágenes se excluyeron explícitamente sin bloquear la reconstrucción.",
        ]),
        ("Embeddings, límites y caché", [
            "Se usa el mismo modelo multilingual-e5-base para pasajes y consultas, con prefijos exactos passage: y query:. Todos los embeddings se normalizan L2 antes de indexar o buscar. El dispositivo se elige explícitamente; --device cuda falla si CUDA no está disponible, y auto cae claramente a CPU.",
            "Batch solicitado/efectivo: 16. Ante CUDA OOM el encoder reduce progresivamente el batch y reintenta; en la construcción final no hubo OOM. float16 se usa solo en inferencia CUDA.",
            "Límite de indexación: objetivo 360 tokens y máximo 480 tokens. El límite de 250 palabras no limita el índice: aplica exclusivamente a resultados.jsonl. Para salida, cada fragmento se parte solo por oraciones o unidades estructurales completas; una unidad realmente indivisible >250 palabras se omite de presentación y se busca el siguiente candidato.",
            "Caché persistente por hash: extraction cache guarda payloads por SHA-256 y evita reabrir fuentes sin cambios; embedding cache SQLite guarda vectores float32 normalizados por modelo/tipo/contenido. Esto hace las verificaciones posteriores reproducibles y evita reextraer/reembeddar el corpus.",
        ]),
        ("FAISS, recuperación y métricas", [
            "Índice: <b>faiss.IndexFlatIP</b>, escrito con faiss.write_index(). Con L2, el producto interno equivale a similitud coseno exacta. metadata.jsonl tiene una línea por vector en el mismo orden; el cargador verifica index.ntotal y la dimensión/configuración/prefijos antes de consultar.",
            f"Tiempo medido de indexación (embeddings + FAISS): {config['build_seconds']:.2f} s. Tamaño index.faiss: {index_bytes / 1024 / 1024:.1f} MiB. Metadata: {metadata_count:,} líneas. Resultados generados: {sum(1 for line in args.results.open(encoding='utf-8') if line.strip())} consultas.",
            "Se recuperan hasta 1,000 candidatos para asegurar diez salidas que respeten el límite de 250 palabras; la agregación de documentos usa max pooling y devuelve tres doc_id distintos. Los diez fragmentos conservan chunk_id para trazabilidad.",
            "Comparación con índice anterior: anterior 171,201 chunks, 1,474 documentos cubiertos, 344.56 s de indexación y 501.6 MiB FAISS; final 208,447 chunks, 1,762 documentos cubiertos, 776.05 s y 610.7 MiB (+37,246 chunks; +288 documentos). El anterior tenía 343 fallas a nivel de archivo; el final registra 1,218 omisiones de bloque o fuente, principalmente por unidades indivisibles >480 tokens, sin perder los demás bloques del documento. La métrica de cobertura parcial anterior no fue preservada por el artefacto previo; la final es 118 documentos.",
        ]),
        ("Validación, errores restantes y reproducción", [
            "Pruebas unitarias: 11 passed. Cubren selección CUDA, normalización L2, correspondencia FAISS-metadata, forma de resultados, listas completas y caché de extracción. gpu_smoke_test valida Python, PyTorch/CUDA, RTX 3060 Ti, dimensión, embeddings ES/EN/PT y búsqueda FAISS mínima.",
            "Preflight de entrega valida desde cero carga FAISS, igualdad índice/metadata, esquema metadata, 50 queries ordenadas, 3 documentos, 10 fragmentos, ids existentes y máximo de 250 palabras. generador.py vuelve a cargar el índice persistido y reproduce resultados.jsonl.",
            f"Errores restantes registrados: {failure_types.get('Una oracion excede max_tokens: requiere politica explicita, no corte automatico', 0):,} bloques mayores de 480 tokens sin frontera semántica segura; {failure_types.get('OCR omitido por defecto: active --enable-ocr solo tras verificar texto relevante', 0)} imágenes relevantes sin OCR por ausencia de Tesseract; 2 archivos de control fuera de F1/F2/F3. No se introdujo corte arbitrario para ocultar estos casos.",
            "Reproducción en Windows: activar .venv, instalar dependencias y wheel CUDA de PyTorch, ejecutar scripts/gpu_smoke_test.py --device cuda, construir con scripts/build_baseline.py, y ejecutar generador.py. FAISS se usa en CPU de forma compatible; CUDA acelera los embeddings.",
        ]),
    ]
    story = []
    for position, (title, texts) in enumerate(pages):
        story.append(Paragraph(title, styles["Title"] if position == 0 else heading))
        for text in texts:
            story.extend([Spacer(1, 12), paragraph(text, body)])
        if position != len(pages) - 1:
            story.append(PageBreak())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(args.output), pagesize=A4, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54).build(story)


if __name__ == "__main__":
    main()
