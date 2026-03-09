# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Pipeline di preprocessing RAG
# ─────────────────────────────────────────────────────────────────────────────
#
# Usiamo Python 3.11 slim (non 3.13) per massima compatibilità con le
# dipendenze — la stessa logica per cui avevamo problemi con unstructured.
# "slim" = immagine base minimale, senza tool non necessari (riduce la size).

FROM python:3.11-slim

# Metadati dell'immagine (opzionali ma utili)
LABEL project="normatrix"
LABEL component="preprocessing"

# ── Sistema operativo ─────────────────────────────────────────────────────────
# Installiamo le dipendenze di sistema necessarie per pdfplumber e sentence-transformers.
# - libgomp1: libreria OpenMP richiesta da torch/transformers per parallelismo CPU
# - curl: usato dall'healthcheck per testare ChromaDB (vedi docker-compose.yml)
# "rm -rf /var/lib/apt/lists/*" pulisce la cache apt dopo l'installazione,
# riducendo la dimensione finale dell'immagine.
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Directory di lavoro ───────────────────────────────────────────────────────
# Tutto avverrà in /app dentro il container.
WORKDIR /app

# ── Dipendenze Python ─────────────────────────────────────────────────────────
# IMPORTANTE: copiamo PRIMA solo requirements.txt e installiamo le dipendenze.
# Poi copiamo il codice. Questo sfrutta la cache di Docker:
# se cambi solo il codice (non requirements.txt), Docker riusa il layer
# delle dipendenze già installate → build molto più veloce.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Codice ───────────────────────────────────────────────────────────────────
# Copiamo tutto il codice. Il .dockerignore (vedi sotto) esclude
# file non necessari come venv, chroma_db, __pycache__, ecc.
COPY *.py .

# ── Variabili d'ambiente default ──────────────────────────────────────────────
# In Docker la pipeline si connette a ChromaDB via HTTP.
# Questi valori vengono sovrascritti da docker-compose.yml se necessario.
ENV CHROMA_MODE=http
ENV CHROMA_HOST=chromadb
ENV CHROMA_PORT=8000

# ── Comando di default ────────────────────────────────────────────────────────
# Può essere sovrascritto in docker-compose.yml o al momento del docker run.
# --input /app/documents → cartella montata come volume in docker-compose.yml
CMD ["python", "pipeline.py", "--input", "/app/documents", "--collection", "bank_policies"]
