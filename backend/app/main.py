"""
main.py
---------
FastAPI application that exposes the driving-school RAG assistant
and serves the frontend chat website from the same server.

Run with:
    uvicorn app.main:app --reload --port 8000

Then open http://localhost:8000 in the browser.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load variables from backend/.env before anything else needs them
# (must happen before importing llm_client, which reads GROQ_API_KEY).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .rag import get_index
from .llm_client import generate_answer
from .stt_client import transcribe_audio

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Building RAG index...")
    get_index()          # هيبني الـ Embeddings مرة واحدة عند تشغيل السيرفر
    print("RAG index ready.")
    yield

app = FastAPI(
    title="Driving School RAG Assistant",
    version="2.0.0",
    lifespan=lifespan,
)
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

class HistoryTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[HistoryTurn] = []


class ChatSource(BaseModel):
    type: str
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]


class VoiceChatResponse(ChatResponse):
    transcript: str  # النص اللي طلع من الملف الصوتي (عشان تعرضه في الواجهة)


# ---------- shared chat logic (used by /api/chat and /api/chat/voice) ----------

def _run_chat(message: str, history: list[HistoryTurn]) -> ChatResponse:
    if not message.strip():
        raise HTTPException(status_code=400, detail="message is empty")

    idx = get_index()
    chunks = idx.search(message, top_k=8)

    history_dicts = [{"role": t.role, "content": t.content} for t in history]

    try:
        answer = generate_answer(message, chunks, history=history_dicts)
    except RuntimeError as e:
        # Most likely: GROQ_API_KEY missing. Return retrieved context
        # anyway so the frontend/dev can still see the RAG step working.
        raise HTTPException(status_code=500, detail=str(e))

    sources = [ChatSource(type=c.doc_type, text=c.text) for c in chunks]
    return ChatResponse(answer=answer, sources=sources)


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
    return _run_chat(req.message, req.history)


@app.post("/api/chat/voice", response_model=VoiceChatResponse)
async def chat_voice(
    file: UploadFile = File(...),
    history: str = Form("[]"),
):
    """
    بيستقبل ملف صوتي (رسالة صوتية من المستخدم)، يحوّله لنص باستخدام
    faster-whisper (محليًا، من غير API خارجي)، وبعدين يمرر النص الناتج
    لنفس الـ RAG + LLM pipeline بتاع /api/chat عشان يرجع رد نصي.

    history: نفس شكل history في /api/chat، لكن مبعوتة كـ JSON string
    جوه الـ form (زي [{"role": "user", "content": "..."}]) لأن
    الطلب multipart/form-data مش JSON body عادي.
    """
    # 1) نحفظ الملف الصوتي مؤقتًا على الديسك عشان faster-whisper يقدر يقرأه
    suffix = Path(file.filename or "").suffix or ".wav"
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 2) نحوّل الصوت لنص
        try:
            transcript = transcribe_audio(tmp_path)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"تعذر تحويل الصوت إلى نص: {e}"
            )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not transcript.strip():
        raise HTTPException(
            status_code=400, detail="لم يتم التعرف على أي كلام في الملف الصوتي"
        )

    # 3) نفك history المبعوتة كـ JSON string (لو موجودة/سليمة)
    try:
        raw_history = json.loads(history) if history else []
        history_turns = [HistoryTurn(**h) for h in raw_history]
    except Exception:
        history_turns = []

    # 4) نفس منطق /api/chat بالظبط، لكن بالنص المستخرج من الصوت
    result = _run_chat(transcript, history_turns)
    return VoiceChatResponse(
        answer=result.answer, sources=result.sources, transcript=transcript
    )


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


@app.get("/api/filters")
def filters():
    """Distinct values to populate dropdown filters in the frontend."""
    idx = get_index()
    return {
        "areas": idx.distinct_areas(),
        "governorates": idx.distinct_governorates(),
        "engine_ccs": idx.distinct_engine_ccs(),
        "service_types": idx.distinct_service_types(),
        "cities": idx.distinct_cities(),
    }


# ---------- serve the frontend ----------

@app.get("/")
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
