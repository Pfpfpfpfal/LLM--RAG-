from __future__ import annotations
print("INGEST: imported annotations")
from pathlib import Path
print("INGEST: imported Path")
import json, hashlib, logging
print("INGEST: imported stdlib")
from typing import List, Dict, Any
print("INGEST: imported typing")
import numpy as np
print("INGEST: imported numpy")
from tqdm import tqdm
print("INGEST: imported tqdm")
from rank_bm25 import BM25Okapi
print("INGEST: imported rank_bm25")
# from sentence_transformers import SentenceTransformer
# print("INGEST: imported sentence_transformers")
# import faiss
# print("INGEST: imported faiss")
from rag.chunk import split_markdown_into_chunks, Chunk
print("INGEST: imported rag.chunk")

print("started ingestion script...")

log_path = Path("data/index/ingest.log")
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(log_path),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info("Logger initialized. CWD=%s", Path.cwd())


def file_hash(file_path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(file_path.read_bytes())
    return hasher.hexdigest()

def simple_tokenizer(text: str) -> List[str]:
    text = text.lower()
    text = text.replace("`", " ").replace("*", " ").replace("#", " ")
    token = []
    cur = []
    for ch in text:
        if ch.isalnum() or ch in ("_",):
            cur.append(ch)
        else:
            if cur:
                token.append("".join(cur))
                cur = []
    if cur:
        token.append("".join(cur))
    return token

def ingest_markdown_files(notes_dir: str = "data/notes", index_dir: str = "data/index"):
    notes_path = Path(notes_dir)
    out = Path(index_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    md_files =sorted(notes_path.rglob("*.md"))
    if not md_files:
        raise RuntimeError(f"No markdown files found in {notes_dir}")
    
    all_chunks: List[Chunk] = []
    file_manifests: Dict[str, Any] = {}
    
    for f in tqdm(md_files, desc="files", unit="file"):
        rel = str(f.relative_to(notes_path)).replace("\\", "/")
        txt = f.read_text(encoding="utf-8", errors="ignore")
        h = file_hash(f)
        file_manifests[rel] = {"sha256": h}
        
        chunks = split_markdown_into_chunks(txt, source_file=rel, max_chars=1200, overlap=100)
        logging.info(f"{rel} chunks={len(chunks)} chars={sum(len(c.text) for c in chunks)}")
        for i, c in enumerate(chunks):
            c.meta["chunk_id"] = f"{rel}::sec{c.meta['section_index']}::ch{c.meta['chunk_in_section']}"
        all_chunks.extend(chunks)
        
        if len(all_chunks) % 200 == 0:
            logging.info("Progress: files processed=%d/%d, total_chunks=%d", md_files.index(f)+1, len(md_files), len(all_chunks))

        
    texts = [c.text for c in all_chunks]
    metas = [c.meta for c in all_chunks]
    
    # tokenized = [simple_tokenizer(t) for t in texts]
    # bm25 = BM25Okapi(tokenized)
    
    # logging.info("Loading embedding model...")
    # model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # logging.info("Model loaded.")
    
    # emb_list = []
    # bs = 64
    # for i in range(0, len(texts), bs):
    #     batch = texts[i:i+bs]
    #     e = model.encode(batch, batch_size=bs, show_progress_bar=True, normalize_embeddings=True)
    #     emb_list.append(np.asarray(e, dtype=np.float32))
    # 
    # emb = np.vstack(emb_list)
    # 
    # dim = emb.shape[1]
    # index = faiss.IndexFlatIP(dim)
    # index.add(emb)
    # 
    # faiss.write_index(index, str(out / "faiss.index"))
    # np.save(out / "embeddings.npy", emb)
    (out / "chunks.jsonl").write_text("\n".join(json.dumps({"text": t, "meta": m}, ensure_ascii=False) for t, m in zip(texts, metas)), encoding="utf-8")
    (out / "manifests.json").write_text(json.dumps(file_manifests, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print(f"Indexed files: {len(md_files)}")
    print(f"Total chunks: {len(all_chunks)}")
    print("Done")
    
if __name__ == "__main__":
    ingest_markdown_files()