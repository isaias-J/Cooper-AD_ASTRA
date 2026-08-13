# CODEFEST AD ASTRA 2026 - Delivery

This folder is the package to submit. Do not upload the full repository or the
corpus. Keep the folder structure intact:

```text
entrega/
    resultados.jsonl
    generador.py
    informe_tecnico.pdf
    base_vectorial/
        encoder_multilingual_e5_base/
            index.faiss
            metadata.jsonl
            encoder_config.json
        grafo/
            grafo.graphml
    src/
        codefest/
```

The `src/` directory is a support dependency for `generador.py`; it makes the
script reproducible when run directly from this folder. `encoder_config.json`
records the encoder, vector dimension, prefixes, and FAISS configuration.

## Run The Delivered System

Install Python 3.11, Git LFS, an NVIDIA driver, and the dependencies:

```powershell
git lfs install
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install faiss-cpu==1.12.0 sentence-transformers==5.1.0 networkx==3.5 numpy
```

From the repository root, use the official JSONL query file supplied by
CODEFEST:

```powershell
python entrega\generador.py `
  --queries C:\ruta\queries_official.jsonl `
  --index-dir entrega\base_vectorial\encoder_multilingual_e5_base `
  --graph entrega\base_vectorial\grafo\grafo.graphml `
  --output entrega\resultados.jsonl `
  --device cuda `
  --batch-size 16
```

The delivered results already contain 50 JSONL records. The graph is the
optional bonus component and contributes metadata-linked entity relations as
non-generative retrieval evidence.

`.gitattributes` and `.gitignore` belong at the repository root for Git LFS and
local-file protection. They are not required inside this delivery folder.
