import json
from pathlib import Path
from codefest.chunking import sentences, word_count
from codefest.core import detect_language
from codefest.validate import validate_results
def test_sentence_split_preserves_terminal_sentence():
 assert sentences("Hola mundo. ¿Como estas? Tudo bem!")==["Hola mundo.","¿Como estas?","Tudo bem!"]
def test_result_validator_accepts_shape(tmp_path):
 rows=[]
 for i in range(1,51): rows.append({"query_id":f"q{i:03d}","documents":[{"rank":j,"doc_id":str(j)} for j in range(1,4)],"fragments":[{"rank":j,"chunk_id":f"c{j}","doc_id":"1","text":"Una oracion."} for j in range(1,11)]})
 p=tmp_path/"r.jsonl"; p.write_text("".join(json.dumps(x)+"\n" for x in rows)); assert validate_results(p)==[]
def test_word_limit(): assert word_count("a b c")==3
def test_language_detection_for_required_languages():
 assert detect_language("La seguridad espacial y la defensa nacional") == "es"
 assert detect_language("The space domain and national security") == "en"
 assert detect_language("A segurança espacial é importante para a defesa nacional") == "pt"
