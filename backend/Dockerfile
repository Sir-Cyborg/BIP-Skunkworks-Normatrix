# Dockerfile — Normatrix Backend (FastAPI)
FROM python:3.11-slim

LABEL project="normatrix"
LABEL component="backend"

RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dipendenze backend
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codice backend
COPY . .

# Variabili d'ambiente default
ENV CHROMA_MODE=http
ENV CHROMA_HOST=chromadb
ENV CHROMA_PORT=8000
ENV PIPELINE_PATH=/app/pipeline_src
ENV DOCUMENTS_DIR=/app/documents
ENV CHROMA_COLLECTION_POLICIES=bank_policies

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]