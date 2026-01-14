from fastapi import FastAPI
from pydantic import BaseModel
from rag.retrieval import RagStore, build_extractive_answer

app = FastAPI(title="Notes RAG (local)")

store = None

class AskReq(BaseModel):
    query: str
    strict: bool = True

@app.on_event("startup")
def load():
    global store
    store = RagStore(index_dir="data/index")

@app.post("/ask")
def ask(req: AskReq):
    retrieved = store.search(req.query, k_vec=10, k_bm25=10, k_final=10)
    answer, cites = build_extractive_answer(req.query, retrieved, strict=req.strict)
    return {
        "answer": answer,
        "citations": cites,
        "retrieved": retrieved[:5],
    }
