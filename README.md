# Cooper - CODEFEST AD ASTRA 2026

Recuperación vectorial pura con un encoder público de Hugging Face, embeddings L2 normalizados y `faiss.IndexFlatIP`. No usa LLM, BM25, ChromaDB, reranking, cross-encoder ni expansión de consultas. El mismo `intfloat/multilingual-e5-base` y sus prefijos E5 (`passage:`/`query:`) se usan al indexar y consultar.

## Windows + RTX 3060 Ti

Requisitos: Windows 10/11, driver NVIDIA funcional y Python 3.11. No es necesario instalar CUDA Toolkit: el wheel de PyTorch trae su runtime CUDA.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Wheel oficial probado para Windows/Pip/CUDA 12.8:
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

VS Code queda configurado para `${workspaceFolder}\\.venv\\Scripts\\python.exe` en `.vscode/settings.json`.

### Verificar GPU

La primera ejecución descarga `intfloat/multilingual-e5-base` desde Hugging Face (aproximadamente 1 GB de caché). No indexa el corpus.

```powershell
nvidia-smi
python scripts/gpu_smoke_test.py --device cuda
```

Debe mostrar `torch.cuda.is_available(): True`, `NVIDIA GeForce RTX 3060 Ti`, dispositivo efectivo `cuda`, embeddings normalizados y una búsqueda `IndexFlatIP` correcta.

## Indexar el corpus

El corpus completo descomprimido ocupa cerca de 3 GB y la indexación puede tardar. Para auditarlo sin cargar el modelo ni crear embeddings:

```powershell
python scripts/build_baseline.py --corpus-root "C:\ruta\CORPUS CODEFEST AD ASTRA 2026" --audit-only
```

Para construir con CUDA y un batch inicial conservador para 8 GB (se reduce automáticamente si ocurre OOM):

```powershell
python scripts/build_baseline.py --corpus-root "C:\ruta\CORPUS CODEFEST AD ASTRA 2026" --device cuda --batch-size 16
```

Se crean `index.faiss`, `metadata.jsonl` y `encoder_config.json` en `base_vectorial/encoder_multilingual_e5_base/`. FAISS funciona en CPU en Windows; la RTX acelera los embeddings. `.cache/` conserva extracción por hash y embeddings normalizados para evitar releer/reembeddar archivos intactos. El máximo de 480 tokens aplica al índice; el límite de 250 palabras aplica solamente a cada fragmento de `resultados.jsonl`. Las imágenes se omiten explícitamente salvo que se habilite OCR verificado con `--enable-ocr`.

### OCR trazable para documentos escaneados

El reglamento recomienda OCR para imágenes, gráficos o PDF sin capa textual cuando contienen información semántica. El OCR se usa únicamente para extraer texto antes de E5; no participa en ranking y no es un modelo generativo. La entrega auditada recupera 48 PDF escaneados y tres figuras analíticas. Dieciocho JSON administrativos y seis imágenes decorativas permanecen excluidos para no contaminar el índice.

Instale Tesseract 5 y los modelos oficiales ES/EN/PT fijados a una revisión reproducible:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_ocr.ps1
python scripts\ocr_smoke_test.py --input "C:\ruta\documento_escaneado.pdf" --tesseract-command "C:\Program Files\Tesseract-OCR\tesseract.exe" --tessdata-dir .cache\tessdata
```

Construcción completa con OCR selectivo:

```powershell
python scripts\build_baseline.py --corpus-root "C:\ruta\CORPUS CODEFEST AD ASTRA 2026" --device cuda --batch-size 16 --enable-ocr --ocr-languages spa+eng+por --ocr-dpi 250 --ocr-min-confidence 60 --tesseract-command "C:\Program Files\Tesseract-OCR\tesseract.exe" --tessdata-dir .cache\tessdata --ocr-image-allowlist config\ocr_image_allowlist.txt
```

Los bloques por debajo de 60/100 de confianza se descartan; también se eliminan cabeceras y pies OCR cortos repetidos dentro del mismo documento. `metadata.jsonl` conserva `ocr_engine`, idiomas, confianza, páginas y cantidad de bloques repetidos eliminados. La clave de caché incluye motor, idiomas, DPI, confianza y hash del archivo.

## Generar resultados

```powershell
python scripts/extract_queries.py --pdf "C:\ruta\Extracto_Preguntas_50_v2.pdf"
python generador.py --queries data/processed/queries_official.jsonl --index-dir base_vectorial/encoder_multilingual_e5_base --output resultados.jsonl --device cuda
```

Si existe `base_vectorial/grafo/grafo.graphml`, el generador lo autodetecta y aplica una fusión conservadora (`--graph-weight 0.0025`). Se puede desactivar para una comparación vectorial pura con `--no-graph`. El peso se seleccionó comparando similitud coseno, diversidad y trazabilidad; no se afirma una mejora de Recall sin ground truth oficial.

El cargador exige que FAISS y metadata tengan el mismo número de registros y que modelo, dimensión y prefijos coincidan con `encoder_config.json`. La salida contiene top-10 chunks y tres documentos distintos. El validador también comprueba que cada `chunk_id` pertenezca al `doc_id` declarado.

## Pruebas y entrega

```powershell
python -m pytest -q
python generador.py --queries data/processed/queries_official.jsonl --index-dir base_vectorial/encoder_multilingual_e5_base --no-graph --output output/resultados_baseline.jsonl --device cuda
python generador.py --queries data/processed/queries_official.jsonl --index-dir base_vectorial/encoder_multilingual_e5_base --output output/resultados_repro.jsonl --device cuda
python scripts/render_technical_report.py --index-dir base_vectorial/encoder_multilingual_e5_base --results resultados.jsonl --baseline-results output/resultados_baseline.jsonl --reproduction-results output/resultados_repro.jsonl --manifest data/processed/documents_manifest.jsonl --graph base_vectorial/grafo/grafo.graphml --output output/informe_tecnico.pdf
python scripts/package_delivery.py --results resultados.jsonl --index-dir base_vectorial/encoder_multilingual_e5_base --graph base_vectorial/grafo/grafo.graphml --report output/informe_tecnico.pdf
python scripts/validate_delivery.py --delivery-dir entrega
```

## Troubleshooting CUDA

- `torch.cuda.is_available()` es `False`: confirme `nvidia-smi`, active `.venv` y ejecute `python -m pip show torch`. Si la versión muestra CPU, reinstale únicamente PyTorch con el comando CUDA anterior.
- `CUDA out of memory`: cierre procesos que usen GPU según `nvidia-smi` o reduzca `--batch-size`. El encoder ya reintenta con la mitad hasta llegar a 1.
- Driver/runtime: `nvidia-smi` muestra la versión máxima soportada por el driver; `python -c "import torch; print(torch.__version__, torch.version.cuda)"` muestra el runtime incluido en PyTorch. No tienen que ser idénticas.
- VS Code usa otro Python: seleccione manualmente `.venv\\Scripts\\python.exe` y compruebe `python -c "import sys; print(sys.executable)"`.
- Diagnóstico completo: `python scripts/gpu_smoke_test.py --device cuda`. Para verificar solo lógica sin GPU: `python scripts/gpu_smoke_test.py --device cpu --no-fp16`.
- `Tesseract no encontrado`: ejecute `scripts\setup_ocr.ps1` y compruebe `& "C:\Program Files\Tesseract-OCR\tesseract.exe" --version`.
- `Failed loading language`: compruebe que `.cache\tessdata` contenga `spa.traineddata`, `eng.traineddata`, `por.traineddata` y `osd.traineddata`; vuelva a ejecutar el script de instalación, sin borrar el índice existente.
