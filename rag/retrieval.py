from __future__ import annotations
from pathlib import Path
import json
from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = text.replace("`", " ").replace("*", " ").replace("#", " ")
    text = text.replace("[[", " ").replace("]]", " ")
    tokens, cur = [], []
    for ch in text:
        if ch.isalnum() or ch == "_":
            cur.append(ch)
        else:
            if cur:
                tokens.append("".join(cur))
                cur = []
    if cur:
        tokens.append("".join(cur))
    return tokens

class RagStore:
    def __init__(
        self,
        index_dir: str = "data/index",
        qdrant_url: str = "http://127.0.0.1:6333",
    ):
        p = Path(index_dir)
        chunks_path = p / "chunks.jsonl"
        if not chunks_path.exists():
            raise RuntimeError("chunks.jsonl not found. Run python -m rag.ingest first.")

        self.chunks: List[Dict[str, Any]] = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]
        self.texts = [c["text"] for c in self.chunks]
        self.metas = [c["meta"] for c in self.chunks]

        tokenized = [simple_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

        model_name = (p / "embed_model.txt").read_text(encoding="utf-8").strip()
        self.embedder = TextEmbedding(model_name=model_name)

        self.collection = (p / "qdrant_collection.txt").read_text(encoding="utf-8").strip()
        self.qdrant = QdrantClient(url=qdrant_url)

    def search(self, query: str, k_vec: int = 8, k_bm25: int = 8, k_final: int = 8):
        q_vec = next(self.embedder.embed([query]))
        q_vec = np.asarray(q_vec, dtype=np.float32)

        q_vec = next(self.embedder.embed([query]))
        q_vec = np.asarray(q_vec, dtype=np.float32)

        if hasattr(self.qdrant, "query_points"):
            qr = self.qdrant.query_points(
                collection_name=self.collection,
                query=q_vec,
                limit=k_vec,
                with_payload=True,
            )
            hits = qr.points
        else:
            hits = self.qdrant.search(
                collection_name=self.collection,
                query_vector=q_vec,
                limit=k_vec,
                with_payload=True,
            )

        vec = [(int(h.id), float(h.score)) for h in hits]

        q_tok = simple_tokenize(query)
        scores_b = self.bm25.get_scores(q_tok)
        idx_b = np.argsort(scores_b)[::-1][:k_bm25]
        bm = [(int(i), float(scores_b[i])) for i in idx_b]

        merged: Dict[int, float] = {}
        for i, s in vec:
            merged[i] = merged.get(i, 0.0) + 1.0 * s

        if bm:
            max_b = max(s for _, s in bm) or 1.0
            for i, s in bm:
                merged[i] = merged.get(i, 0.0) + 0.6 * (s / max_b)

        top = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:k_final]
        res = []
        for i, s in top:
            res.append({"score": s, "text": self.texts[i], "meta": self.metas[i]})
        return res

def build_extractive_answer(query: str, retrieved: List[Dict[str, Any]], strict: bool = True):
    if not retrieved:
        return ("Ничего не найдено по этому вопросу", [])

    if strict and retrieved[0]["score"] < 0.05:
        return ("Не нашёл достаточной информации, чтобы уверенно ответить", [])

    best = retrieved[0]["score"]
    filtered = [r for r in retrieved if r["score"] >= 0.75 * best]
    use = filtered[:3]
    cites = []
    parts = []

    for r in use:
        meta = r["meta"]
        hp_list = meta.get("header_path", [])
        hp = " > ".join(hp_list) if hp_list else "(без заголовка)"
        parts.append(r["text"])
        cites.append(
            {
                "source_file": meta.get("source_file"),
                "header_path": hp,
                "chunk_id": meta.get("chunk_id"),
            }
        )

    return ("\n\n".join(parts).strip(), cites)