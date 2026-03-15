"""
retriever.py - Recupero evidenze da ChromaDB per ogni domanda requisito

Dato una domanda (es. "L'organizzazione ha definito procedure di notifica?")
cerca i top-k chunk più rilevanti nella collection bank_policies.
"""

import os
import time
from loguru import logger
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ── Config da environment ─────────────────────────────────────────────────────
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
CHROMA_MODE = os.getenv("CHROMA_MODE", "embedded")
CHROMA_DIR  = os.getenv("CHROMA_DIR", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION_POLICIES", "bank_policies")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
TOP_K = int(os.getenv("RETRIEVER_TOP_K", 5))


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVER
# ─────────────────────────────────────────────────────────────────────────────

class Retriever:
    """
    Cerca in ChromaDB i chunk più rilevanti per una domanda.
    Singleton — il modello di embedding viene caricato una sola volta.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self):
        if self._initialized:
            return

        logger.info(f"Caricamento modello embedding: {EMBEDDING_MODEL}")
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Modello embedding pronto.")

        # Connessione ChromaDB con retry
        retries = int(os.getenv("CHROMA_WAIT_RETRIES", 15))
        interval = int(os.getenv("CHROMA_WAIT_INTERVAL", 5))

        for attempt in range(1, retries + 1):
            try:
                if CHROMA_MODE == "http":
                    self._client = chromadb.HttpClient(
                        host=CHROMA_HOST,
                        port=CHROMA_PORT,
                        settings=Settings(anonymized_telemetry=False),
                    )
                else:
                    self._client = chromadb.PersistentClient(
                        path=CHROMA_DIR,
                        settings=Settings(anonymized_telemetry=False),
                    )
                self._client.heartbeat()
                logger.info(f"Connesso a ChromaDB al tentativo {attempt}")
                break
            except Exception as e:
                if attempt == retries:
                    raise RuntimeError(f"Impossibile connettersi a ChromaDB: {e}")
                logger.warning(f"ChromaDB non pronto ({attempt}/{retries}), riprovo tra {interval}s...")
                time.sleep(interval)

        self._initialized = True

    def get_collection(self, collection_name: str = CHROMA_COLLECTION):
        """Restituisce una collection ChromaDB, creandola se non esiste."""
        try:
            return self._client.get_collection(collection_name)
        except Exception:
            return None

    def list_collections(self) -> list[dict]:
        """
        Lista tutte le collection con nome e numero di chunk.
        Usato dalla pagina Corpus Management.
        """
        try:
            collections = self._client.list_collections()
            result = []
            for col in collections:
                try:
                    c = self._client.get_collection(col.name)
                    result.append({"name": col.name, "count": c.count()})
                except Exception:
                    result.append({"name": col.name, "count": 0})
            return result
        except Exception as e:
            logger.error(f"Errore list_collections: {e}")
            return []

    def search(
        self,
        question: str,
        collection_name: str = CHROMA_COLLECTION,
        top_k: int = TOP_K,
    ) -> list[dict]:
        """
        Cerca i top-k chunk più rilevanti per una domanda.

        Returns:
            Lista di dict con: text, score, source, section_path, page_number
        """
        col = self.get_collection(collection_name)
        if col is None or col.count() == 0:
            logger.warning(f"Collection '{collection_name}' vuota o non esistente.")
            return []

        embedding = self._model.encode(question).tolist()

        results = col.query(
            query_embeddings=[embedding],
            n_results=min(top_k, col.count()),
            include=["documents", "metadatas", "distances"],
        )

        evidences = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            evidences.append({
                "text": doc,
                "score": round(1 - dist, 4),           # distanza coseno → similarità
                "source": meta.get("source", "N/A"),
                "section_path": meta.get("section_path", "N/A"),
                "page_number": meta.get("page_number", "N/A"),
                "has_table": meta.get("has_table", False),
            })

        return evidences


# Singleton globale
retriever = Retriever()