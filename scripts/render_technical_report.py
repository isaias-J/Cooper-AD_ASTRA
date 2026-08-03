#!/usr/bin/env python3
"""Generate the required <=8-page technical PDF after an index has been built."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

def main():
 p=argparse.ArgumentParser(); p.add_argument("--index-dir",type=Path,required=True); p.add_argument("--output",type=Path,default=Path("output/informe_tecnico.pdf")); a=p.parse_args()
 cfg=json.loads((a.index_dir/"encoder_config.json").read_text(encoding="utf-8")); a.output.parent.mkdir(parents=True,exist_ok=True)
 styles=getSampleStyleSheet(); body=styles["BodyText"]; story=[Paragraph("Informe técnico - Cooper | CODEFEST AD ASTRA 2026",styles["Title"])]
 sections=[
 ("Alcance", "Base de conocimiento vectorial para recuperación. No se utilizan LLM, query expansion, BM25, ChromaDB ni reranking generativo."),
 ("Preprocesamiento y chunking", "Un archivo original corresponde a un documento. PDF preserva orden y elimina encabezados/pies repetidos; JSON extrae campos textuales; CSV/XLSX usa pares columna: valor por fila; PBF usa atributos y deduplicación intra-tile. Los cortes se hacen únicamente al final de una oración. Tamaño objetivo: 360 tokens; máximo: 480 tokens."),
 ("Encoder", f"Modelo: {cfg['model']}. Dimensión: {cfg['dimension']}. Prefijos E5: passage para chunks y query para consultas. Embeddings normalizados L2. Candidatos no declarados ganadores sin ground truth: multilingual-e5-large y BAAI/bge-m3."),
 ("Índice y recuperación", f"FAISS IndexFlatIP con vectores L2, equivalente a coseno exacto. Chunks indexados: {cfg['chunks']}. Construcción: {cfg['build_seconds']} s. Se recuperan 100 candidatos, top-10 fragmentos y tres documentos mediante max pooling."),
 ("Reproducibilidad y validación", "generador.py carga el índice persistido y produce resultados.jsonl. validate_delivery.py verifica FAISS-metadata, metadata obligatoria y el esquema de 50 consultas. Los fragmentos de salida no exceden 250 palabras ni cortan oraciones."),
 ]
 for title,text in sections: story += [Spacer(1,10),Paragraph(title,styles["Heading2"]),Paragraph(text,body)]
 SimpleDocTemplate(str(a.output),pagesize=A4,leftMargin=54,rightMargin=54,topMargin=54,bottomMargin=54).build(story)
if __name__=="__main__": main()
