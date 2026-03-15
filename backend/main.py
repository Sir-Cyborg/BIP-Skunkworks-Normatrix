"""
main.py - FastAPI backend per Normatrix

Endpoints:
  GET  /                          → serve il frontend (index.html)
  GET  /api/corpus                → lista documenti nel DB
  POST /api/corpus/upload         → carica PDF e triggera pipeline
  POST /api/analysis/upload       → carica CSV/Excel con domande
  POST /api/analysis/run          → esegue retrieval + valutazione
  GET  /api/health                → health check
"""

import io
import os
import sys
import tempfile
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from loguru import logger

import pandas as pd

# Aggiungi il path della pipeline al sys.path per importare i moduli esistenti
PIPELINE_PATH = os.getenv("PIPELINE_PATH", "/app/pipeline_src")
sys.path.insert(0, PIPELINE_PATH)

from retriever import retriever
from evaluator import evaluator

# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Normatrix API",
    description="RAG pipeline per compliance documentale",
    version="1.0.0",
)

# Serve i file statici (frontend)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Inizializza il retriever all'avvio del server."""
    logger.info("Inizializzazione Normatrix backend...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, retriever.initialize)
    logger.info("Backend pronto.")


# ─────────────────────────────────────────────────────────────────────────────
# FRONTEND
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "normatrix-backend"}


# ─────────────────────────────────────────────────────────────────────────────
# CORPUS MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/corpus")
async def get_corpus():
    """
    Restituisce la lista delle collection ChromaDB con il numero di chunk.
    Usato dalla pagina Corpus Management per mostrare i documenti indicizzati.
    """
    try:
        collections = retriever.list_collections()
        return {"collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/corpus/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Riceve un PDF, lo salva nella cartella documents e triggera la pipeline.
    La pipeline gira in background — la risposta è immediata.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo file PDF sono accettati.")

    # Salva il PDF nella cartella documents condivisa con la pipeline
    documents_dir = Path(os.getenv("DOCUMENTS_DIR", "/app/documents"))
    documents_dir.mkdir(parents=True, exist_ok=True)
    dest = documents_dir / file.filename

    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    logger.info(f"PDF salvato: {dest}")

    # Triggera la pipeline in background
    background_tasks.add_task(run_pipeline, str(dest))

    return {
        "status": "ok",
        "message": f"'{file.filename}' caricato. Indicizzazione in corso...",
        "filename": file.filename,
    }


def run_pipeline(pdf_path: str):
    """Esegue la pipeline di preprocessing in background."""
    try:
        from pipeline import PreprocessingPipeline
        pipeline = PreprocessingPipeline(
            collection_name=os.getenv("CHROMA_COLLECTION_POLICIES", "bank_policies")
        )
        stats = pipeline.run(pdf_path)
        logger.info(f"Pipeline completata: {stats}")
    except Exception as e:
        logger.error(f"Errore pipeline: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

class Question(BaseModel):
    id: str
    question: str


class AnalysisRequest(BaseModel):
    questions: list[Question]
    collection: str = "bank_policies"
    top_k: int = 5


@app.post("/api/analysis/upload")
async def upload_questions(file: UploadFile = File(...)):
    """
    Riceve un CSV o Excel con colonne ID e Question.
    Restituisce la lista di domande parsate.
    """
    filename = file.filename.lower()
    if not (filename.endswith(".csv") or filename.endswith(".xlsx") or filename.endswith(".xls")):
        raise HTTPException(
            status_code=400,
            detail="Formato non supportato. Usa CSV o Excel (.xlsx, .xls)."
        )

    content = await file.read()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Errore lettura file: {e}")

    # Normalizza nomi colonne (case insensitive)
    df.columns = [c.strip().lower() for c in df.columns]

    if "id" not in df.columns or "question" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Il file deve avere le colonne 'ID' e 'Question'. Trovate: {list(df.columns)}"
        )

    questions = [
        {"id": str(row["id"]), "question": str(row["question"])}
        for _, row in df.iterrows()
        if str(row["question"]).strip()
    ]

    logger.info(f"Caricate {len(questions)} domande da {file.filename}")
    return {"questions": questions, "total": len(questions)}


@app.post("/api/analysis/run")
async def run_analysis(request: AnalysisRequest):
    """
    Per ogni domanda:
      1. Retriever cerca top-k chunk in ChromaDB
      2. Evaluator (placeholder) valuta compliance

    Restituisce risultati completi + statistiche dashboard.
    """
    if not request.questions:
        raise HTTPException(status_code=400, detail="Nessuna domanda fornita.")

    results = []
    compliant = 0
    non_compliant = 0
    partial = 0

    for q in request.questions:
        # 1. Retrieval
        evidences = retriever.search(
            question=q.question,
            collection_name=request.collection,
            top_k=request.top_k,
        )

        # 2. Valutazione (placeholder)
        evaluation = evaluator.evaluate(
            question=q.question,
            evidences=evidences,
        )

        # Contatori dashboard
        if evaluation.status == "COMPLIANT":
            compliant += 1
        elif evaluation.status == "NON_COMPLIANT":
            non_compliant += 1
        else:
            partial += 1

        results.append({
            "id": q.id,
            "question": q.question,
            "evidences": evidences,
            "evaluation": {
                "status": evaluation.status,
                "explanation": evaluation.explanation,
                "confidence": evaluation.confidence,
                "is_placeholder": evaluation.is_placeholder,
            },
        })

    total = len(results)
    return {
        "results": results,
        "dashboard": {
            "total": total,
            "compliant": compliant,
            "non_compliant": non_compliant,
            "partial": partial,
            "compliance_rate": round(compliant / total * 100, 1) if total > 0 else 0,
        },
    }