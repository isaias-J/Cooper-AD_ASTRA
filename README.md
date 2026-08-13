# CODEFEST AD ASTRA 2026

This repository is organized in two branches:

- `main` / `entrega_final`: final delivery package.
- `implementacion-completa`: complete source code, tests, build scripts, and corpus workflow.

Git does not allow spaces in branch names, so `implementacion-completa` is the valid form of `implementacion completa`.

## Final Delivery

The folder to submit is `entrega/`. It contains the required JSONL results,
reproducible generator, technical PDF, FAISS index, metadata, and optional
GraphML bonus. The `src/` directory inside `entrega/` is included because the
generator imports it when executed from the delivery folder.

The root `.gitattributes` is required by Git LFS for the large FAISS and
metadata files. The root `.gitignore` prevents local corpora, caches, models,
and generated files from being committed. Keep both files at repository root.

## Continue On NVIDIA PC

Clone the delivery branch with Git LFS:

```powershell
git lfs install
git clone -b entrega_final https://github.com/isaias-J/Cooper-AD_ASTRA.git
cd Cooper-AD_ASTRA
```

Use Python 3.11 and an NVIDIA driver:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install faiss-cpu==1.12.0 sentence-transformers==5.1.0 networkx==3.5 numpy
```

Verify CUDA and the downloaded LFS files:

```powershell
nvidia-smi
git lfs pull
python -c "import torch; print(torch.cuda.is_available())"
```

## Reproduce Results

The official query file is supplied separately by the organizers. It must
contain exactly `q001` through `q050` in JSONL format. From the repository root:

```powershell
python entrega/generador.py `
  --queries C:\ruta\queries_official.jsonl `
  --index-dir entrega\base_vectorial\encoder_multilingual_e5_base `
  --graph entrega\base_vectorial\grafo\grafo.graphml `
  --output entrega\resultados.jsonl `
  --device cuda `
  --batch-size 16
```

The graph is optional bonus evidence. It uses deterministic entity and
relation extraction and does not use a generative model.

## Complete Implementation Workflow

For rebuilding the index from the full corpus, switch to the complete branch:

```powershell
git switch implementacion-completa
git pull
```

Keep the extracted corpus outside Git, preferably at the repository root with
the name `CORPUS CODEFEST AD ASTRA 2026`:

```powershell
python scripts/build_baseline.py `
  --corpus-root "C:\ruta\CORPUS CODEFEST AD ASTRA 2026" `
  --device cuda `
  --batch-size 16
```

Build the bonus graph:

```powershell
python scripts/build_knowledge_graph.py `
  --metadata base_vectorial\encoder_multilingual_e5_base\metadata.jsonl `
  --output base_vectorial\grafo\grafo.graphml `
  --min-mentions 3
```

Generate graph-aware results, render the report, package, and validate:

```powershell
python generador.py --queries data\processed\queries_official.jsonl --index-dir base_vectorial\encoder_multilingual_e5_base --graph base_vectorial\grafo\grafo.graphml --output resultados.jsonl --device cuda
python scripts/render_technical_report.py --index-dir base_vectorial\encoder_multilingual_e5_base --results resultados.jsonl --output output\informe_tecnico.pdf
python scripts/package_delivery.py --results resultados.jsonl --index-dir base_vectorial\encoder_multilingual_e5_base --graph base_vectorial\grafo\grafo.graphml --report output\informe_tecnico.pdf
python scripts/validate_delivery.py --delivery-dir entrega
```

The final validator must print `PRECHECK PASSED`.

## Tests

```powershell
python -m pytest -q
python -m compileall -q src scripts generador.py tests
```

Do not commit the corpus, `.cache`, model files, FAISS build outputs, Python
caches, or temporary reports. The root `.gitignore` excludes these artifacts.
