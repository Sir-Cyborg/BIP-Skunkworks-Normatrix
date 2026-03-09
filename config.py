"""
config.py - Configurazione centralizzata per la pipeline di preprocessing RAG
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"          # PDF originali
OUTPUT_DIR = BASE_DIR / "data" / "processed"  # JSON intermedi (opzionale)
CHROMA_DIR = BASE_DIR / "chroma_db"           # Persistenza ChromaDB (solo modalità embedded)

# ─────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DIM = 768

# ─────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────
CHUNK_STRATEGY = "hierarchical"

CHUNK_SIZES = {
    "large": 1024,
    "medium": 512,
    "small": 256,
}
CHUNK_OVERLAP = 32
RETRIEVAL_CHUNK_LEVEL = "medium"

# ─────────────────────────────────────────────
# CHROMADB — modalità di connessione
# ─────────────────────────────────────────────
CHROMA_COLLECTION_POLICIES = "bank_policies"
CHROMA_COLLECTION_REGULATIONS = "regulations"

# Modalità di connessione:
#   "embedded" → ChromaDB gira dentro Python, dati in CHROMA_DIR (default locale)
#   "http"     → ChromaDB gira come server separato (Docker)
#
# In locale non serve impostare nulla → usa "embedded" automaticamente.
# In Docker, docker-compose.yml imposta CHROMA_MODE=http tramite variabile d'ambiente.
CHROMA_MODE = os.getenv("CHROMA_MODE", "embedded")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))

# ─────────────────────────────────────────────
# PARSING
# ─────────────────────────────────────────────
MIN_TEXT_LENGTH = 30