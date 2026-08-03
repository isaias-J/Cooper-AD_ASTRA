# Informe técnico - CODEFEST AD ASTRA 2026

Este documento se convierte a PDF tras la indexación. Debe mantenerse por debajo de ocho páginas.

## Diseño

Base vectorial multilingüe construida exclusivamente con encoders públicos de Hugging Face y FAISS. No se emplearon modelos generativos, BM25, ChromaDB ni reranking generativo.

## Preprocesamiento y chunking

Se preserva un documento por archivo original. PDFs conservan orden de lectura y eliminan cabeceras/pies repetitivos; JSON selecciona campos textuales; CSV/XLSX representan cada fila como pares `columna: valor`; PBF convierte atributos a texto y deduplica elementos del tile. Los chunks respetan límites oracionales completos y usan el tokenizer del encoder para el conteo.

## Encoder e índice

Encoder inicial: `intfloat/multilingual-e5-base`, con prefijos `passage:` y `query:`. Vectores normalizados L2 y `faiss.IndexFlatIP`, equivalente a similitud coseno exacta. Completar después de ejecutar: dimensión, chunks, tiempo, hardware y memoria.

## Recuperación

Se recuperan 100 candidatos; se devuelven los 10 mejores fragmentos y tres documentos diferentes mediante max pooling por `doc_id`. Fragmentos de salida se dividen solo por oraciones para mantener el límite de 250 palabras.

## Validación y reproducibilidad

`generador.py` vuelve a cargar el índice sin reindexar. `scripts/validate_delivery.py` verifica la alineación FAISS-metadata y el JSONL final.
