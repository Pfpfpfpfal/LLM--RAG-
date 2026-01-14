from __future__ import annotations
from pathlib import Path
import json
from typing import List, Dict, Any, Tuple
# import numpy as np
# import faiss
from rank_bm25 import BM25Okapi
# from sentence_transformers import SentenceTransformer

def simple_tokenize(text: str) -> List[str]:
    text = text.lower().replace("`", " ").replace("*", " ").replace("#", " ")
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
    def __init__(self, index_dir: str = "data/index"):
        p = Path(index_dir)
        self.chunks: List[Dict[str, Any]] = []
        for line in (p / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
            self.chunks.append(json.loads(line))

        self.texts = [c["text"] for c in self.chunks]
        self.metas = [c["meta"] for c in self.chunks]

        # self.emb = np.load(p / "embeddings.npy").astype(np.float32)
        # self.faiss = faiss.read_index(str(p / "faiss.index"))

        # self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        tokenized = [simple_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k_bm25: int = 8) -> List[Dict[str, Any]]:
        q_tok = simple_tokenize(query)
        scores_b = self.bm25.get_scores(q_tok)
        
        res = []
        for i in sorted(range(len(scores_b)), key=lambda i: scores_b[i], reverse=True)[:k_bm25]:
            res.append({
                "score": float(scores_b[i]),
                "text": self.texts[i],
                "meta": self.metas[i],
            })
        return res

def build_extractive_answer(query: str, retrieved: List[Dict[str, Any]], strict: bool = True) -> Tuple[str, List[Dict[str, Any]]]:
    if not retrieved:
        return ("Ничего не найдено по этому вопросу", [])

    if strict and retrieved[0]["score"] < 0.2:
        return ("Не нашёл достаточной информации", [])

    use = retrieved[:3]
    parts = []
    cites = []
    for r in use:
        meta = r["meta"]
        hp = " > ".join(meta.get("header_path", []))
        parts.append(r["text"])
        cites.append({
            "source_file": meta.get("source_file"),
            "header_path": hp,
            "chunk_id": meta.get("chunk_id"),
        })

    answer = "\n\n".join(parts).strip()
    return answer, cites