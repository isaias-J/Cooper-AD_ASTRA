"""Deterministic multilingual entity/relation graph for the optional bonus."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

KNOWN_ENTITIES = (
    "inteligencia artificial", "seguridad espacial", "órbita baja terrestre",
    "cambio climático", "derechos humanos", "américa latina", "américa del sur",
    "estados unidos", "unión europea", "fuerzas armadas", "fuerza aeroespacial",
    "sector defensa", "ciencia y tecnología", "desarrollo humano", "space debris",
    "low earth orbit", "artificial intelligence", "machine learning", "cybersecurity",
)
STOP_ENTITIES = {
    "El", "La", "Los", "Las", "Un", "Una", "Uno", "The", "This", "That", "These",
    "Para", "Como", "Desde", "Entre", "Según", "También", "Sin", "Sobre", "Más",
}
ENTITY_RE = re.compile(r"\b(?:[A-ZÁÉÍÓÚÜÑÇÃÕ][\wÁÉÍÓÚÜÑáéíóúüñçãõ-]*)(?:\s+(?:[A-ZÁÉÍÓÚÜÑÇÃÕ][\wÁÉÍÓÚÜÑáéíóúüñçãõ-]*)){0,5}\b")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
RELATION_RE = re.compile(
    r"\b(desarrolla|desarrollan|utiliza|utilizan|emplea|emplean|afecta|amenaza|"
    r"regula|regulan|fortalece|fortalecen|depende|depender|protege|protegen|"
    r"promueve|promueven|impacta|impactan|genera|generan|requiere|requieren|"
    r"cooper[aá]|contribuye|contribuyen|enfrenta|enfrentan|improves?|supports?|"
    r"threatens?|regulates?|affects?|uses?|develops?)\b",
    re.IGNORECASE,
)


def normalize_entity(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip(" ,;:()[]{}\"'"))
    return unicodedata.normalize("NFC", value).lower()


def extract_entities(text: str) -> list[tuple[str, str]]:
    """Return unique (canonical label, coarse type) pairs from ES/EN/PT text."""
    found: dict[str, str] = {}
    lowered = text.lower()
    for phrase in KNOWN_ENTITIES:
        start = 0
        while True:
            position = lowered.find(phrase, start)
            if position < 0:
                break
            canonical = normalize_entity(text[position:position + len(phrase)])
            found.setdefault(canonical, "concept")
            start = position + len(phrase)
    for match in ENTITY_RE.finditer(text):
        label = normalize_entity(match.group(0))
        if label and label.title() not in STOP_ENTITIES and len(label) > 2:
            entity_type = "organization" if any(token in label for token in ("universidad", "ministerio", "nato", "onu", "fuerza")) else "entity"
            found.setdefault(label, entity_type)
    return sorted(found.items())


def extract_relations(text: str, entities: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """Extract typed relations only from sentences containing an explicit trigger."""
    labels = {label for label, _ in entities}
    relations = []
    for sentence in SENTENCE_RE.split(text):
        normalized_sentence = normalize_entity(sentence)
        sentence_entities = [(label, normalized_sentence.find(label)) for label in labels if label in normalized_sentence]
        trigger = RELATION_RE.search(sentence)
        if trigger and len(sentence_entities) >= 2:
            before = [item for item in sentence_entities if item[1] < trigger.start()]
            after = [item for item in sentence_entities if item[1] >= trigger.end()]
            subject = max(before, key=lambda item: item[1])[0] if before else sentence_entities[0][0]
            object_ = min(after, key=lambda item: item[1])[0] if after else sentence_entities[-1][0]
            if subject != object_:
                relations.append((subject, trigger.group(1).lower(), object_))
    return sorted(set(relations))


def build_graph(metadata_records, min_entity_mentions: int = 1) -> nx.DiGraph:
    graph = nx.DiGraph(name="CODEFEST AD ASTRA knowledge graph")
    mentions = Counter()
    evidence: defaultdict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
    types: dict[str, str] = {}
    for record in metadata_records:
        entities = extract_entities(record.get("texto", ""))
        for label, entity_type in entities:
            mentions[label] += 1
            types.setdefault(label, entity_type)
        for subject, relation, object_ in extract_relations(record.get("texto", ""), entities):
            evidence[(subject, relation, object_)].append((record["doc_id"], record["chunk_id"]))
    included = {label for label, count in mentions.items() if count >= min_entity_mentions}
    active_labels = {label for triple in evidence for label in (triple[0], triple[2]) if label in included}
    for label in active_labels:
        graph.add_node(f"entity:{label}", label=label, entity_type=types[label], mentions=str(mentions[label]))
    for (subject, relation, object_), records in evidence.items():
        if subject not in included or object_ not in included:
            continue
        graph.add_edge(
            f"entity:{subject}",
            f"entity:{object_}",
            relation=relation,
            evidence_count=str(len(records)),
            doc_ids=";".join(sorted({doc_id for doc_id, _ in records})),
            chunk_ids=";".join(sorted({chunk_id for _, chunk_id in records})),
        )
    return graph


def write_graph(metadata_path: Path, output_path: Path, min_entity_mentions: int = 1) -> dict:
    import json

    with metadata_path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    graph = build_graph(records, min_entity_mentions=min_entity_mentions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, output_path)
    return {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(), "path": str(output_path)}


def graph_chunk_scores(graph: nx.Graph, query: str) -> Counter:
    """Score indexed chunks linked to entities mentioned in a query."""
    query_entities = {label for label, _ in extract_entities(query)}
    scores = Counter()
    if not query_entities:
        return scores
    for source, target, attributes in graph.edges(data=True):
        source_label = normalize_entity(str(graph.nodes[source].get("label", source)))
        target_label = normalize_entity(str(graph.nodes[target].get("label", target)))
        if source_label not in query_entities and target_label not in query_entities:
            continue
        weight = int(attributes.get("evidence_count", "1"))
        for chunk_id in str(attributes.get("chunk_ids", "")).split(";"):
            if chunk_id:
                scores[chunk_id] += weight
    return scores
