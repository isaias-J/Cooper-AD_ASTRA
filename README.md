# CODEFEST AD ASTRA 2026

Este repositorio está organizado en dos ramas:

- `main` / `entrega_final`: paquete final para entregar.
- `implementacion-completa`: código fuente completo, pruebas, scripts y flujo de construcción.

Git no permite espacios en los nombres de ramas. Por eso se usa
`implementacion-completa` en lugar de `implementacion completa`.

## Entrega Final

La carpeta que se debe entregar es `entrega/`. Contiene los resultados JSONL,
el generador reproducible, el informe técnico, el índice FAISS, la metadata y
el grafo GraphML del bonus.

El directorio `src/` dentro de `entrega/` se incluye porque `generador.py` lo
necesita para ejecutarse directamente desde la carpeta de entrega.

El archivo `.gitattributes` en la raíz es necesario para Git LFS, que gestiona
los archivos grandes del índice FAISS y la metadata. El archivo `.gitignore`
evita subir corpus, cachés, modelos y archivos generados localmente. Ambos
deben permanecer en la raíz del repositorio.

## Continuar En Un PC Con NVIDIA

Clona la rama de entrega usando Git LFS:

```powershell
git lfs install
git clone -b entrega_final https://github.com/isaias-J/Cooper-AD_ASTRA.git
cd Cooper-AD_ASTRA
```

Usa Python 3.11 y un controlador NVIDIA instalado:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install faiss-cpu==1.12.0 sentence-transformers==5.1.0 networkx==3.5 numpy
```

Verifica CUDA y los archivos descargados mediante Git LFS:

```powershell
nvidia-smi
git lfs pull
python -c "import torch; print(torch.cuda.is_available())"
```

## Reproducir Resultados

El archivo oficial de consultas es suministrado por los organizadores. Debe
contener exactamente `q001` hasta `q050` en formato JSONL. Desde la raíz:

```powershell
python entrega/generador.py `
  --queries C:\ruta\queries_official.jsonl `
  --index-dir entrega\base_vectorial\encoder_multilingual_e5_base `
  --graph entrega\base_vectorial\grafo\grafo.graphml `
  --output entrega\resultados.jsonl `
  --device cuda `
  --batch-size 16
```

El grafo es el componente bonus opcional. Usa extracción determinista de
entidades y relaciones, sin modelos generativos.

## Flujo De Implementación Completa

Para reconstruir el índice desde el corpus completo, cambia a la rama de
implementación:

```powershell
git switch implementacion-completa
git pull
```

Mantén el corpus descomprimido fuera de Git, preferiblemente en la raíz del
repositorio con el nombre `CORPUS CODEFEST AD ASTRA 2026`:

```powershell
python scripts/build_baseline.py `
  --corpus-root "C:\ruta\CORPUS CODEFEST AD ASTRA 2026" `
  --device cuda `
  --batch-size 16
```

Construye el grafo bonus:

```powershell
python scripts/build_knowledge_graph.py `
  --metadata base_vectorial\encoder_multilingual_e5_base\metadata.jsonl `
  --output base_vectorial\grafo\grafo.graphml `
  --min-mentions 3
```

Genera resultados usando el grafo, crea el informe, empaqueta y valida:

```powershell
python generador.py --queries data\processed\queries_official.jsonl --index-dir base_vectorial\encoder_multilingual_e5_base --graph base_vectorial\grafo\grafo.graphml --output resultados.jsonl --device cuda
python scripts/render_technical_report.py --index-dir base_vectorial\encoder_multilingual_e5_base --results resultados.jsonl --output output\informe_tecnico.pdf
python scripts/package_delivery.py --results resultados.jsonl --index-dir base_vectorial\encoder_multilingual_e5_base --graph base_vectorial\grafo\grafo.graphml --report output\informe_tecnico.pdf
python scripts/validate_delivery.py --delivery-dir entrega
```

El validador debe mostrar `PRECHECK PASSED`.

## Pruebas

```powershell
python -m pytest -q
python -m compileall -q src scripts generador.py tests
```

No subas el corpus, `.cache`, modelos, índices generados, cachés de Python ni
reportes temporales. El `.gitignore` de la raíz excluye esos archivos.
