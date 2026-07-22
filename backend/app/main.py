"""
main.py
---------
FastAPI application that exposes the driving-school RAG assistant
and serves the frontend chat website from the same server.

Run with:
    uvicorn app.main:app --reload --port 8000

Then open http://localhost:8000 in the browser.
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .rag import get_index
from .llm_client import generate_answer

app = FastAPI(title="Driving School RAG Assistant", version="1.0.0")

# Allow the frontend (served from anywhere, including a different port
# during development) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


# ---------- request / response models ----------

class ChatRequest(BaseModel):
    message: str


class ChatSource(BaseModel):
    type: str
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]


# ---------- API routes ----------

@app.get("/api/health")
def health():
    idx = get_index()
    return {
        "status": "ok",
        "license_docs": len(idx.license_docs),
        "schools": len(idx.schools),
        "maintenance_rows": len(idx.maintenance),
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message is empty")

    idx = get_index()
    chunks = idx.search(req.message, top_k=6)

    try:
        answer = generate_answer(req.message, chunks)
    except RuntimeError as e:
        # Most likely: GROQ_API_KEY missing. Return retrieved context
        # anyway so the frontend/dev can still see the RAG step working.
        raise HTTPException(status_code=500, detail=str(e))

    sources = [ChatSource(type=c.doc_type, text=c.text) for c in chunks]
    return ChatResponse(answer=answer, sources=sources)


@app.get("/api/schools")
def schools(area: str = "", governorate: str = ""):
    idx = get_index()
    results = idx.filter_schools(area=area, governorate=governorate)
    return [r.__dict__ for r in results]


@app.get("/api/maintenance")
def maintenance(engine_cc: str = "", service_type: str = "", city: str = ""):
    idx = get_index()
    results = idx.filter_maintenance(
        engine_cc=engine_cc, service_type=service_type, city=city
    )
    return [r.__dict__ for r in results]


# ---------- serve the frontend ----------

@app.get("/")
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
