# Continuation Guide

## Project State

This repository contains the CODEFEST AD ASTRA 2026 Stage 1 retrieval pipeline.
The source implementation and tests are ready. A real FAISS index and official
results have not been committed because they are generated artifacts.

The corpus is local-only and must not be committed. Keep the extracted corpus
in a directory named `CORPUS CODEFEST AD ASTRA 2026` at the repository root,
or pass its absolute path to the build script.

## NVIDIA Machine Setup

Use Python 3.11 on Windows with an NVIDIA driver. Create and activate a virtual
environment, then install the CUDA PyTorch wheel before project dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Verify CUDA before indexing:

```powershell
nvidia-smi
python scripts/gpu_smoke_test.py --device cuda
```

## Build And Deliver

Run the audit first:

```powershell
python scripts/build_baseline.py --corpus-root "C:\path\CORPUS CODEFEST AD ASTRA 2026" --audit-only
```

Build the baseline index. The default model is
`intfloat/multilingual-e5-base`, a multilingual encoder compatible with the
specification:

```powershell
python scripts/build_baseline.py --corpus-root "C:\path\CORPUS CODEFEST AD ASTRA 2026" --device cuda --batch-size 16
```

Generate official results:

```powershell
python generador.py --queries data/processed/queries_official.jsonl --index-dir base_vectorial/encoder_multilingual_e5_base --output resultados.jsonl --device cuda
```

Render the technical report:

```powershell
python scripts/render_technical_report.py --index-dir base_vectorial/encoder_multilingual_e5_base --results resultados.jsonl --output output/informe_tecnico.pdf
```

Package and validate the delivery:

```powershell
python scripts/package_delivery.py --results resultados.jsonl --index-dir base_vectorial/encoder_multilingual_e5_base --report output/informe_tecnico.pdf
python scripts/validate_delivery.py --delivery-dir entrega
```

The final preflight must print `PRECHECK PASSED`.

## Tests

Run before and after the GPU build:

```powershell
python -m pytest -q
python -m compileall -q src scripts generador.py tests
```

## Commit Rules

Do not commit the corpus, `.cache`, model files, FAISS indexes, generated
results, generated reports, Python caches, or extraction failure logs. The
`.gitignore` already excludes these artifacts. Commit source code, tests,
documentation, and configuration only.

The optional knowledge graph and multiple-encoder fusion are not implemented;
they are bonus features and are not required for the baseline delivery.
