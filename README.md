# Cooper - CODEFEST AD ASTRA 2026

Baseline reproducible de recuperacion vectorial pura: mismo encoder multilingue para pasajes y consultas, prefijos E5, embeddings L2 normalizados y `faiss.IndexFlatIP`. No incluye LLM, BM25, ChromaDB, reranking ni expansion de consultas.

Repositorio oficial del equipo Cooper para la Hackathon de Recuperación de Información en Defensa. Incluye investigación, desarrollo y experimentación con búsqueda semántica, embeddings, recuperación de información y técnicas de ranking.

## Preparacion (Windows + RTX 3060 Ti)

```powershell
py --version
nvidia-smi
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Seleccionar el comando CUDA vigente de https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

La ultima comprobacion debe mostrar `True` y `NVIDIA GeForce RTX 3060 Ti` antes de una indexacion competitiva.

## Construccion

```powershell
python scripts/build_baseline.py --corpus-root "C:\ruta\CORPUS CODEFEST AD ASTRA 2026"
python scripts/extract_queries.py --pdf "C:\ruta\Extracto_Preguntas_50_v2.pdf"
python generador.py --queries data/processed/queries_official.jsonl --index-dir base_vectorial/encoder_multilingual_e5_base --output resultados.jsonl
pytest -q
```

El constructor deja `index.faiss`, `metadata.jsonl` y `encoder_config.json` bajo `base_vectorial/encoder_multilingual_e5_base/`. La posicion de cada metadata coincide exactamente con el ID interno de FAISS. `metadata.jsonl` conserva el texto original, sin resumen.

PBF e imagenes se reportan como fallas explicitamente hasta habilitar un extractor de esquema geoespacial u OCR validado; nunca se indexa contenido inventado.
