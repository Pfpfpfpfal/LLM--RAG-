from fastapi import FastAPI
from pydantic import BaseModel
from rag.retrieval import RagStore, build_extractive_answer
from rag.llm import format_context, generate_with_ollama

app = FastAPI(title="Notes RAG (local)")

store = None

class AskReq(BaseModel):
    query: str
    strict: bool = True
    mode: str = "llm"

@app.on_event("startup")
def load():
    global store
    store = RagStore(index_dir="data/index")

@app.post("/ask")
def ask(req: AskReq):
    retrieved = store.search(req.query, k_vec=10, k_bm25=10, k_final=10)
    answer, citations = build_extractive_answer(req.query, retrieved, strict=req.strict)

    if req.mode == "extractive":
        return {"answer": answer, "citations": citations, "retrieved": retrieved[:5]}

    ctx = format_context(retrieved[:5], max_chars=6000)
    llm_answer = generate_with_ollama(req.query, ctx)

    return {"answer": llm_answer, "citations": citations, "retrieved": retrieved[:5]}