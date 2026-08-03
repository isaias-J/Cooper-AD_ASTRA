# Informe de implementación - Cooper

## Estado de esta ejecución

La ruta de corpus verificada fue `/Users/cris/Downloads/CORPUS CODEFEST AD ASTRA 2026`. La auditoría produjo 1.839 documentos: 459 del fenómeno 1, 479 del fenómeno 2, 899 del fenómeno 3 y 2 sin fenómeno (los archivos de control ubicados en la raíz). Los formatos presentes son 964 JSON, 760 PDF, 73 PBF, 26 CSV, 6 XLSX, 8 JPG, 1 AVIF y 1 TXT. No se hallaron archivos vacíos.

`data/processed/documents_manifest.jsonl` contiene una línea por archivo y 1.839 IDs deterministas derivados de la ruta relativa normalizada. `data/processed/queries_official.jsonl` contiene exactamente q001 a q050, extraídas del PDF proporcionado sin reformulación.

## Implementado

- Inventario JSON/Markdown y manifest estable.
- Extractores modulares para PDF (PyMuPDF), JSON, CSV, XLSX, TXT/MD y HTML.
- PBF real mediante `mapbox-vector-tile`, con atributos convertidos a texto y deduplicación intra-tile; imágenes excluidas por defecto y OCR opcional con Tesseract tras revisar relevancia.
- Chunking por oraciones completas, conteo mediante tokenizer del encoder y chunks objetivo de 360 tokens (máximo 480 configurable).
- Encoder configurable con `intfloat/multilingual-e5-base`, prefijos `passage:`/`query:`, normalización L2 y candidatos documentados (`multilingual-e5-large`, `bge-m3`).
- `faiss.IndexFlatIP`, persistencia `index.faiss`, metadata JSONL alineada por orden de inserción, recuperación top-100/top-10 y agregación documental configurable.
- CLI `generador.py` y validadores del formato de entrega.

## Entorno y bloqueo de indexación

Esta máquina de desarrollo es macOS, Python 3.11.15; `nvidia-smi` no está disponible y PyTorch/FAISS/sentence-transformers/PyMuPDF aún no están instalados. Por ello no se descargó el modelo de embeddings ni se generó un índice falso, embeddings simulados o resultados oficiales. En la estación Windows con RTX 3060 Ti se debe seguir el README, verificar CUDA y ejecutar `scripts/build_baseline.py`; el constructor registra GPU, dimensión, tiempos y número final de chunks en `encoder_config.json`.

## Validación ejecutada

- Auditoría real: correcta (1.839 manifest entries).
- Extracción de consultas: correcta (50 líneas, q001--q050).
- Compilación Python y smoke tests de segmentación, límite de palabras y esquema de resultados: correctos.
- `pytest` en `.venv`: correcto tras corregir dos defectos detectados por la prueba (signo de apertura de interrogación en la segmentación ES y fixture incompleto del esquema de fragmentos).

## Riesgos y decisiones pendientes

- Se probaron extractores reales de JSON, PDF, CSV, XLSX y PBF contra una muestra del corpus. Los documentos sin fenómeno (dos archivos de control de raíz) se excluyen del índice para respetar el campo obligatorio `fenomeno`.
- Antes de activar OCR, se debe confirmar que las 9 imágenes contienen texto relevante; la implementación conserva el motor de OCR como metadata.
- No se declara un encoder ganador sin métricas con consultas de desarrollo o ground truth. La baseline usa el encoder inicial exigido, no una afirmación de superioridad.
