"""Persistent content-addressed caches for extraction and normalized embeddings."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ExtractionCache:
    """Map an unchanged source to a payload keyed by its content hash."""

    def __init__(self, root: Path):
        self.root = root / "extractions"
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        try:
            self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self.index = {}

    def get(self, path: Path, cache_key: str):
        stat = path.stat()
        entry = self.index.get(cache_key)
        if entry and entry["size"] == stat.st_size and entry["mtime_ns"] == stat.st_mtime_ns:
            payload = self.root / f"{entry['sha256']}.json"
            if payload.exists():
                return json.loads(payload.read_text(encoding="utf-8")), True
        return None, False

    def put(self, path: Path, cache_key: str, payload: dict) -> str:
        digest = sha256_file(path)
        destination = self.root / f"{digest}.json"
        if not destination.exists():
            destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        stat = path.stat()
        self.index[cache_key] = {"sha256": digest, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        self.index_path.write_text(json.dumps(self.index, ensure_ascii=False, indent=2), encoding="utf-8")
        return digest


class EmbeddingCache:
    """SQLite cache: one normalized float32 vector per model/kind/content hash."""

    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(root / "embeddings.sqlite3")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "model TEXT NOT NULL, kind TEXT NOT NULL, content_hash TEXT NOT NULL, "
            "dimension INTEGER NOT NULL, vector BLOB NOT NULL, "
            "PRIMARY KEY(model, kind, content_hash))"
        )

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_many(self, model: str, kind: str, values: list[str]):
        hashes = [self._hash(value) for value in values]
        found: dict[str, np.ndarray] = {}
        for start in range(0, len(hashes), 900):
            selected = hashes[start:start + 900]
            marks = ",".join("?" for _ in selected)
            rows = self.connection.execute(
                f"SELECT content_hash, dimension, vector FROM embeddings WHERE model=? AND kind=? AND content_hash IN ({marks})",
                [model, kind, *selected],
            )
            for content_hash, dimension, vector in rows:
                found[content_hash] = np.frombuffer(vector, dtype=np.float32, count=dimension).copy()
        return hashes, found

    def put_many(self, model: str, kind: str, values: list[str], vectors: np.ndarray) -> None:
        rows = [
            (model, kind, self._hash(value), int(vector.size), np.asarray(vector, dtype=np.float32).tobytes())
            for value, vector in zip(values, vectors, strict=True)
        ]
        self.connection.executemany(
            "INSERT OR REPLACE INTO embeddings(model, kind, content_hash, dimension, vector) VALUES(?,?,?,?,?)", rows
        )
        self.connection.commit()
