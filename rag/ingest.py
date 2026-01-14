from __future__ import annotations
from pathlib import Path
import json
import hashlib
import logging
from typing import Dict, Any, List
from tqdm import tqdm
import numpy as np
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
import fastembed, qdrant_client
from qdrant_client.models import VectorParams, Distance, PointStruct
from rag.chunk import split_markdown_into_chunks, Chunk

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()

def ingest_markdown_files(
    notes_dir: str = "data/notes",
    index_dir: str = "data/index",
    qdrant_url: str = "http://127.0.0.1:6333",
    collection_name: str = "notes_chunks",
):
    out = Path(index_dir)
    out.mkdir(parents=True, exist_ok=True)

    log_path = out / "ingest.log"
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("Logger initialized. CWD=%s", Path.cwd())

    notes_path = Path(notes_dir)
    md_files = sorted(notes_path.rglob("*.md"))
    if not md_files:
        raise RuntimeError(f"No .md files found in {notes_dir}")

    all_chunks: List[Chunk] = []
    file_manifest: Dict[str, Any] = {}

    logging.info("Chunking %d files...", len(md_files))
    for f in tqdm(md_files, desc="files", unit="file"):
        rel = str(f.relative_to(notes_path)).replace("\\", "/")
        txt = f.read_text(encoding="utf-8", errors="ignore")
        file_manifest[rel] = {"sha256": file_hash(f)}

        chunks = split_markdown_into_chunks(txt, source_file=rel, max_chars=1200, overlap=100)
        for c in chunks:
            c.meta["chunk_id"] = f"{rel}::sec{c.meta['section_index']}::ch{c.meta['chunk_in_section']}"
        all_chunks.extend(chunks)

        logging.info("%s chunks=%d chars=%d", rel, len(chunks), sum(len(c.text) for c in chunks))

    texts = [c.text for c in all_chunks]
    metas = [c.meta for c in all_chunks]

    (out / "chunks.jsonl").write_text(
        "\n".join(json.dumps({"text": t, "meta": m}, ensure_ascii=False) for t, m in zip(texts, metas)),
        encoding="utf-8",
    )
    (out / "manifests.json").write_text(json.dumps(file_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    logging.info("Total chunks: %d", len(texts))

    embed_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    logging.info("fastembed=%s qdrant-client=%s model=%s", fastembed.__version__, qdrant_client.__version__, embed_model)
    embedder = TextEmbedding(model_name=embed_model)

    first_vec = next(embedder.embed([texts[0]])).astype(np.float32)
    dim = int(first_vec.shape[0])
    logging.info("Embedding model: %s (dim=%d)", embed_model, dim)

    client = QdrantClient(url=qdrant_url)

    logging.info("Recreating Qdrant collection: %s", collection_name)
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    BATCH = 128
    points: List[PointStruct] = []

    def flush_points():
        nonlocal points
        if points:
            client.upsert(collection_name=collection_name, points=points)
            points = []

    logging.info("Uploading vectors to Qdrant...")
    for i, vec in enumerate(tqdm(embedder.embed(texts), total=len(texts), desc="fastembed", unit="chunk")):
        v = np.asarray(vec, dtype=np.float32)
        payload = {
            "source_file": metas[i].get("source_file"),
            "header_path": metas[i].get("header_path", []),
            "chunk_id": metas[i].get("chunk_id"),
            "section_index": metas[i].get("section_index"),
            "chunk_in_section": metas[i].get("chunk_in_section"),
        }
        points.append(PointStruct(id=i, vector=v, payload=payload))
        if len(points) >= BATCH:
            flush_points()

    flush_points()

    (out / "embed_model.txt").write_text(embed_model, encoding="utf-8")
    (out / "qdrant_collection.txt").write_text(collection_name, encoding="utf-8")
    logging.info("Done. Indexed files=%d, chunks=%d", len(md_files), len(texts))
    print(f"Indexed files: {len(md_files)}")
    print(f"Total chunks: {len(texts)}")
    print("Done")

if __name__ == "__main__":
    ingest_markdown_files()