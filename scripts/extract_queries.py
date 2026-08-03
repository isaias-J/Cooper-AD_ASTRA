#!/usr/bin/env python3
"""Extracts the supplied 50 questions; does not manufacture queries."""
import argparse, json, re, subprocess
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument("--pdf",type=Path,required=True); p.add_argument("--output",type=Path,default=Path("data/processed/queries_official.jsonl")); a=p.parse_args()
 try:
  text=subprocess.check_output(["pdftotext",str(a.pdf),"-"],text=True)
 except FileNotFoundError:
  import fitz
  document=fitz.open(a.pdf)
  try: text="\n".join(page.get_text("text",sort=True) for page in document)
  finally: document.close()
 matches=re.findall(r"(?ms)\b(q\d{3})\s+(.*?)(?=\s*q\d{3}\s|\Z)",text)
 rows=[{"query_id":qid,"query":" ".join(body.split())} for qid,body in matches]
 if [x["query_id"] for x in rows] != [f"q{i:03d}" for i in range(1,51)]: raise ValueError(f"Se esperaban q001-q050; se obtuvieron {len(rows)}")
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
if __name__=="__main__": main()
