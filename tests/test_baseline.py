import json
import numpy as np
from pathlib import Path
from codefest.chunking import sentences, word_count, chunk_text
from codefest.core import detect_language
from codefest.validate import validate_results
from codefest.vector import l2_normalize, select_device
from codefest.retrieval import Retriever
from codefest.cache import ExtractionCache
from codefest.extract import _json_blocks, _ocr_image_blocks, remove_repeated_ocr_blocks
from codefest.graph import build_graph, extract_entities, graph_chunk_scores, build_graph_evidence_index, indexed_graph_chunk_scores
def test_sentence_split_preserves_terminal_sentence():
 assert sentences("Hola mundo. ¿Como estas? Tudo bem!")==["Hola mundo.","¿Como estas?","Tudo bem!"]
def test_sentence_split_preserves_closing_quote():
 assert sentences('Primera oración.” Segunda oración.') == ['Primera oración.”', 'Segunda oración.']
def test_list_items_are_never_cut():
 text="- Primer ítem completo.\n  Continuación del primer ítem.\n- Segundo ítem completo."
 assert sentences(text)==["- Primer ítem completo. Continuación del primer ítem.","- Segundo ítem completo."]
def test_result_validator_accepts_shape(tmp_path):
 rows=[]
 for i in range(1,51): rows.append({"query_id":f"q{i:03d}","documents":[{"rank":j,"doc_id":str(j)} for j in range(1,4)],"fragments":[{"rank":j,"chunk_id":f"c{j}","doc_id":"1","text":"Una oracion."} for j in range(1,11)]})
 p=tmp_path/"r.jsonl"; p.write_text("".join(json.dumps(x)+"\n" for x in rows)); assert validate_results(p)==[]
def test_word_limit(): assert word_count("a b c")==3
def test_chunking_optional_word_limit():
 class Tokenizer:
  def encode(self,text,add_special_tokens=False): return text.split()
 text="Uno dos tres. Cuatro cinco seis. Siete ocho nueve."
 assert all(word_count(chunk)<=3 for chunk in chunk_text(text,Tokenizer(),target_tokens=99,max_tokens=99,max_words=3))
 assert len(chunk_text(text,Tokenizer(),target_tokens=99,max_tokens=99))==1
def test_language_detection_for_required_languages():
 assert detect_language("La seguridad espacial y la defensa nacional") == "es"
 assert detect_language("The space domain and national security") == "en"
 assert detect_language("A segurança espacial é importante para a defesa nacional") == "pt"
def test_device_selection():
 assert select_device("auto",True)=="cuda"
 assert select_device("auto",False)=="cpu"
 assert select_device("cpu",True)=="cpu"
 import pytest
 with pytest.raises(RuntimeError,match="solicitó CUDA"): select_device("cuda",False)
def test_l2_normalization():
 vectors=l2_normalize(np.array([[3,4],[5,12]],dtype=np.float32))
 assert vectors.dtype==np.float32
 assert np.allclose(np.linalg.norm(vectors,axis=1),1.0,atol=1e-6)
def test_faiss_metadata_alignment(tmp_path):
 import faiss
 vectors=l2_normalize(np.array([[1,0],[0,1],[1,1]],dtype=np.float32))
 index=faiss.IndexFlatIP(2); index.add(vectors)
 faiss.write_index(index,str(tmp_path/"index.faiss"))
 metadata=[{"chunk_id":f"c{i}"} for i in range(3)]
 (tmp_path/"metadata.jsonl").write_text("".join(json.dumps(x)+"\n" for x in metadata),encoding="utf-8")
 loaded=faiss.read_index(str(tmp_path/"index.faiss"))
 lines=[x for x in (tmp_path/"metadata.jsonl").read_text(encoding="utf-8").splitlines() if x]
 assert loaded.ntotal==len(lines)==3
def test_retriever_rejects_model_mismatch(tmp_path):
 import faiss, pytest
 index=faiss.IndexFlatIP(2); index.add(np.array([[1,0]],dtype=np.float32)); faiss.write_index(index,str(tmp_path/"index.faiss"))
 (tmp_path/"metadata.jsonl").write_text(json.dumps({"doc_id":"d","chunk_id":"c","texto":"x"})+"\n",encoding="utf-8")
 (tmp_path/"encoder_config.json").write_text(json.dumps({"model":"expected","dimension":2,"prefixes":{"passage":"passage: ","query":"query: "}}),encoding="utf-8")
 class Dummy: model_name="other"
 with pytest.raises(ValueError,match="Encoder incompatible"): Retriever(tmp_path,Dummy())
def test_retriever_rejects_negative_graph_weight(tmp_path):
 import faiss, pytest
 index=faiss.IndexFlatIP(2); index.add(np.array([[1,0]],dtype=np.float32)); faiss.write_index(index,str(tmp_path/"index.faiss"))
 (tmp_path/"metadata.jsonl").write_text(json.dumps({"doc_id":"d","chunk_id":"c","texto":"x"})+"\n",encoding="utf-8")
 class Dummy:
  model_name="unused"
  def encode(self,*args,**kwargs): return np.array([[1,0]],dtype=np.float32)
 retriever=Retriever(tmp_path,Dummy())
 with pytest.raises(ValueError,match="graph_weight"): retriever.search("x",graph_weight=-0.1)
def test_extraction_cache_reuses_unchanged_file(tmp_path):
 source=tmp_path/"a.txt"; source.write_text("contenido",encoding="utf-8")
 cache=ExtractionCache(tmp_path/"cache")
 payload={"blocks":[{"text":"contenido","metadata":{}}],"metadata":{}}
 cache.put(source,"a.txt",payload)
 restored, hit=cache.get(source,"a.txt")
 assert hit and restored==payload

def test_extraction_cache_separates_ocr_variants(tmp_path):
 source=tmp_path/"scan.pdf"; source.write_bytes(b"same-content")
 cache=ExtractionCache(tmp_path/"cache")
 native={"blocks":[],"metadata":{"ocr_applied":False}}
 ocr={"blocks":[{"text":"Texto recuperado.","metadata":{"ocr_applied":True}}],"metadata":{}}
 cache.put(source,"scan.pdf",native)
 cache.put(source,"scan.pdf",ocr,variant="ocr-spa-250")
 assert cache.get(source,"scan.pdf")[0]==native
 assert cache.get(source,"scan.pdf",variant="ocr-spa-250")[0]==ocr

def test_ocr_groups_lines_and_filters_low_confidence(tmp_path,monkeypatch):
 import pytesseract
 command=tmp_path/"tesseract.exe"; command.write_bytes(b"")
 data={
  "text":["Texto","válido","ruido"],"conf":["95","85","20"],
  "block_num":[1,1,2],"par_num":[1,1,1],"line_num":[1,1,1],
 }
 monkeypatch.setattr(pytesseract,"image_to_data",lambda *args,**kwargs:data)
 blocks=_ocr_image_blocks(object(),languages="spa",command=command,tessdata_dir=None,psm=3,min_confidence=60)
 assert [text for text,_ in blocks]==["Texto válido"]
 assert blocks[0][1]["ocr_confidence"]==90.0

def test_repeated_ocr_boilerplate_is_removed_without_touching_body():
 blocks=[]
 for page in range(1,7):
  blocks.extend([
   {"text":f"Página {page} de 6","metadata":{"ocr_engine":"tesseract-5","page_start":page,"page_end":page}},
   {"text":f"Contenido analítico exclusivo de la página {page}.","metadata":{"ocr_engine":"tesseract-5","page_start":page,"page_end":page}},
  ])
 result=remove_repeated_ocr_blocks({"blocks":blocks,"metadata":{}})
 assert len(result["blocks"])==6
 assert all("Contenido analítico" in block["text"] for block in result["blocks"])
 assert result["metadata"]["removed_repeated_ocr_blocks"]==6

def test_json_paragraph_arrays_are_extracted_in_order():
    value={"title":"Titulo", "body_paragraphs":["Primero.", "Segundo."], "nested":{"text":"Tercero."}}
    blocks=list(_json_blocks(value))
    assert [text for text, _ in blocks] == ["title: Titulo", "body_paragraphs: Primero.", "body_paragraphs: Segundo.", "text: Tercero."]
    assert blocks[2][1]["json_path"] == "$.body_paragraphs[1]"

def test_metadata_validator_rejects_invalid_required_types(tmp_path):
    metadata=tmp_path/"metadata.jsonl"
    metadata.write_text(json.dumps({
        "doc_id":"d", "chunk_id":"c", "fuente":"source.pdf", "formato":"pdf",
        "fenomeno":"1", "posicion":0, "num_tokens":1, "texto":"Texto."
    })+"\n", encoding="utf-8")
    from codefest.validate import validate_metadata
    assert any("fenomeno invalido" in error for error in validate_metadata(metadata))

def test_bonus_graph_keeps_typed_relation_and_chunk_traceability():
    records=[{"doc_id":"DOC-1", "chunk_id":"DOC-1-chunk-0000", "texto":"Colombia desarrolla inteligencia artificial para la defensa."}]
    graph=build_graph(records)
    assert "inteligencia artificial" in {label for label, _ in extract_entities(records[0]["texto"])}
    assert graph.number_of_nodes() >= 2
    assert graph.number_of_edges() >= 1
    edge=next(iter(graph.edges(data=True)))[2]
    assert edge["relation"] == "desarrolla"
    assert edge["chunk_ids"] == "DOC-1-chunk-0000"
    assert graph_chunk_scores(graph, "Como Colombia usa inteligencia artificial")

def test_graph_query_filters_question_words_and_prefers_specific_entities():
    labels={label for label,_ in extract_entities("¿Cómo usan los Estados Unidos inteligencia artificial?")}
    assert "cómo" not in labels and "estados" not in labels
    assert {"estados unidos", "inteligencia artificial"}.issubset(labels)

def test_indexed_graph_scores_match_reference_implementation():
    records=[
        {"doc_id":"DOC-1","chunk_id":"C-1","texto":"Colombia desarrolla inteligencia artificial para defensa."},
        {"doc_id":"DOC-2","chunk_id":"C-2","texto":"Colombia utiliza inteligencia artificial en operaciones."},
    ]
    graph=build_graph(records)
    query="Colombia e inteligencia artificial"
    assert indexed_graph_chunk_scores(build_graph_evidence_index(graph),query)==graph_chunk_scores(graph,query)

def test_result_validator_rejects_chunk_document_mismatch(tmp_path):
    rows=[]
    for i in range(1,51):
        rows.append({"query_id":f"q{i:03d}","documents":[{"rank":j,"doc_id":str(j)} for j in range(1,4)],"fragments":[{"rank":j,"chunk_id":f"c{j}","doc_id":"wrong","text":"Texto."} for j in range(1,11)]})
    path=tmp_path/"results.jsonl"; path.write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf-8")
    assert any("no corresponde" in error for error in validate_results(path,chunk_to_doc={f"c{i}":"right" for i in range(1,11)}))
