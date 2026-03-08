"""
config.py - Configurazione centralizzata per la pipeline di preprocessing RAG
"""

from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"          # PDF originali
OUTPUT_DIR = BASE_DIR / "data" / "processed"  # JSON intermedi (opzionale)
CHROMA_DIR = BASE_DIR / "chroma_db"           # Persistenza ChromaDB

# ─────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────
# Modello multilingue ottimale per italiano/inglese su documenti formali
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DIM = 768

# ─────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────
CHUNK_STRATEGY = "hierarchical"   # "hierarchical" | "sentence_window" | "fixed"

# Dimensioni chunk per la strategia gerarchica (in token)
CHUNK_SIZES = {
    "large": 1024,    # Livello 1: sezione intera (contesto ampio)
    "medium": 512,    # Livello 2: paragrafo (usato per retrieval)
    "small": 256,     # Livello 3: alzato da 128 per evitare conflitti con metadata
}
CHUNK_OVERLAP = 32   # Ridotto per evitare che overlap + metadata superino chunk size

# Livello di chunk usato per il retrieval (gli altri servono come contesto)
RETRIEVAL_CHUNK_LEVEL = "medium"

# ─────────────────────────────────────────────
# CHROMADB
# ─────────────────────────────────────────────
CHROMA_COLLECTION_POLICIES = "bank_policies"      # Corpus aziendale
CHROMA_COLLECTION_REGULATIONS = "regulations"     # Regolamenti (DORA, ecc.)

# ─────────────────────────────────────────────
# PARSING (Unstructured)
# ─────────────────────────────────────────────
# Strategia Unstructured: "fast" | "hi_res" | "auto"
# - fast: solo testo, nessuna OCR, veloce
# - hi_res: layout detection + OCR, lento ma accurato (richiede detectron2)
# - auto: sceglie automaticamente in base al PDF
UNSTRUCTURED_STRATEGY = "auto"

# Elementi Unstructured da mantenere (filtra il rumore)
ELEMENTS_TO_KEEP = [
    "Title",
    "NarrativeText",
    "ListItem",
    "Table",
    "Header",
    "Footer",      # Tenuto per metadati, poi filtrato se è solo numero pagina
    "FigureCaption",
]

# Lunghezza minima testo per considerare un elemento valido (evita frammenti)
MIN_TEXT_LENGTH = 30