# CODEFEST AD ASTRA 2026 - Entrega

Esta carpeta es el paquete que se debe entregar. No subas el repositorio
completo ni el corpus. Conserva esta estructura:

```text
entrega/
    README.md
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

El directorio `src/` es una dependencia de soporte de `generador.py`; permite
ejecutar el script directamente desde esta carpeta. `encoder_config.json`
registra el encoder, la dimensión de los vectores, los prefijos y la
configuración de FAISS.

## Ejecutar La Entrega

Instala Python 3.11, Git LFS, un controlador NVIDIA y las dependencias:

```powershell
git lfs install
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install faiss-cpu==1.12.0 sentence-transformers==5.1.0 networkx==3.5 numpy
```

Desde la raíz del repositorio, usa el archivo JSONL de consultas oficiales
suministrado por CODEFEST:

```powershell
python entrega\generador.py `
  --queries C:\ruta\queries_official.jsonl `
  --index-dir entrega\base_vectorial\encoder_multilingual_e5_base `
  --graph entrega\base_vectorial\grafo\grafo.graphml `
  --output entrega\resultados.jsonl `
  --device cuda `
  --batch-size 16
```

Los resultados entregados ya contienen 50 registros JSONL. El grafo es el
componente bonus opcional y aporta relaciones entre entidades vinculadas con
documentos y fragmentos, sin utilizar modelos generativos.

`.gitattributes` y `.gitignore` deben permanecer en la raíz del repositorio:
`.gitattributes` permite descargar correctamente los archivos grandes mediante
Git LFS y `.gitignore` protege contra la inclusión accidental del corpus y
archivos temporales. No es necesario copiarlos dentro de esta carpeta.
