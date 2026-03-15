"""
vector_store.py - Embedding e persistenza su ChromaDB

Responsabilità:
  - Inizializzare il modello di embedding (sentence-transformers multilingue)
  - Gestire le collection ChromaDB (policies vs regulations)
  - Indicizzare i chunk (upsert idempotente)
  - Esporre un metodo di ricerca per il retriever
"""

import os
import time
from pathlib import Path
from typing import Optional
from loguru import logger

import chromadb
from chromadb.config import Settings
from chromadb import Collection

from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from preprocessing.config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    CHROMA_COLLECTION_POLICIES,
    CHROMA_COLLECTION_REGULATIONS,
    CHROMA_MODE,
    CHROMA_HOST,
    CHROMA_PORT,
)


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingModel:
    """
    Wrapper sul modello sentence-transformers via LlamaIndex.
    Singleton: il modello viene caricato una volta sola.
    """
    _instance: Optional["EmbeddingModel"] = None

    def __new__(cls, model_name: str = EMBEDDING_MODEL):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.info(f"Caricamento modello embedding: {model_name}")
            cls._instance._model = HuggingFaceEmbedding(
                model_name=model_name,
                trust_remote_code=True,
            )
            logger.info("Modello embedding pronto.")
        return cls._instance

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embedding batch di testi."""
        return self._model._model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embedding di una singola query."""
        return self._model._model.encode(query, convert_to_numpy=True).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# VECTOR STORE
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    """
    Gestisce la persistenza dei chunk su ChromaDB.

    Ogni chunk viene salvato con:
      - embedding vettoriale
      - testo originale
      - metadati filtrabili (source, section_path, chunk_level, page_number, ...)
    """

    def __init__(
        self,
        collection_name: str = CHROMA_COLLECTION_POLICIES,
        persist_dir: str | Path = CHROMA_DIR,
        embedding_model_name: str = EMBEDDING_MODEL,
        chroma_mode: str = CHROMA_MODE,
        chroma_host: str = CHROMA_HOST,
        chroma_port: int = CHROMA_PORT,
    ):
        self.collection_name = collection_name

        # ── Connessione ChromaDB ──────────────────────────────────────────────
        # "embedded": ChromaDB gira dentro Python — utile in locale senza Docker
        # "http":     ChromaDB gira come server separato — usato in Docker
        #
        # Il valore viene letto da config.py che a sua volta legge la variabile
        # d'ambiente CHROMA_MODE. In locale non serve impostare nulla (default embedded).
        # In Docker, docker-compose.yml imposta CHROMA_MODE=http automaticamente.

        if chroma_mode == "http":
            logger.info(f"ChromaDB modalità HTTP | {chroma_host}:{chroma_port}")
            # ── Retry loop ───────────────────────────────────────────────────
            # In Docker, la pipeline parte pochi secondi dopo ChromaDB.
            # ChromaDB potrebbe non essere ancora pronto ad accettare connessioni.
            # Ritentiamo ogni CHROMA_WAIT_INTERVAL secondi fino a CHROMA_WAIT_RETRIES
            # tentativi, poi solleviamo eccezione.
            retries = int(os.getenv("CHROMA_WAIT_RETRIES", 15))
            interval = int(os.getenv("CHROMA_WAIT_INTERVAL", 5))

            for attempt in range(1, retries + 1):
                try:
                    self._client = chromadb.HttpClient(
                        host=chroma_host,
                        port=chroma_port,
                        settings=Settings(anonymized_telemetry=False),
                    )
                    self._client.heartbeat()  # verifica connessione reale
                    logger.info(f"  Connesso a ChromaDB al tentativo {attempt}")
                    break
                except Exception as e:
                    if attempt == retries:
                        raise RuntimeError(
                            f"Impossibile connettersi a ChromaDB dopo {retries} tentativi. "
                            f"Ultimo errore: {e}"
                        )
                    logger.warning(
                        f"  ChromaDB non ancora pronto (tentativo {attempt}/{retries}), "
                        f"riprovo tra {interval}s..."
                    )
                    time.sleep(interval)
        else:
            logger.info(f"ChromaDB modalità embedded | {persist_dir}")
            self.persist_dir = Path(persist_dir)
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )

        # Collection (crea se non esiste)
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # distanza coseno per NLP
        )

        # Modello di embedding (singleton)
        self._embedder = EmbeddingModel(embedding_model_name)

        logger.info(
            f"VectorStore pronto | collection='{collection_name}' "
            f"| elementi esistenti={self._collection.count()}"
        )

    # ── Indicizzazione ────────────────────────────────────────────────────────

    def index_nodes(
        self,
        nodes: list[BaseNode],
        batch_size: int = 64,
        only_leaf: bool = True,
    ) -> int:
        """
        Vettorizza e salva i nodi su ChromaDB.

        Args:
            nodes       : lista di BaseNode (output del Chunker)
            batch_size  : dimensione batch per l'embedding
            only_leaf   : se True, indicizza solo i leaf node (raccomandato
                          per retrieval). I root node servono solo come contesto.

        Returns:
            Numero di chunk effettivamente indicizzati.
        """
        if only_leaf:
            from llama_index.core.node_parser import get_leaf_nodes
            target_nodes = get_leaf_nodes(nodes)
            logger.info(
                f"Indicizzazione leaf node: {len(target_nodes)} / {len(nodes)} totali"
            )
        else:
            target_nodes = nodes

        if not target_nodes:
            logger.warning("Nessun nodo da indicizzare.")
            return 0

        # Filtra nodi già presenti (upsert idempotente)
        existing_ids = set(self._collection.get(ids=[n.node_id for n in target_nodes])["ids"])
        new_nodes = [n for n in target_nodes if n.node_id not in existing_ids]

        if not new_nodes:
            logger.info("Tutti i chunk sono già presenti in ChromaDB. Skip.")
            return 0

        logger.info(f"  Nuovi chunk da indicizzare: {len(new_nodes)}")

        # Batch embedding + upsert
        indexed = 0
        for i in range(0, len(new_nodes), batch_size):
            batch = new_nodes[i : i + batch_size]
            texts = [n.text for n in batch]
            ids = [n.node_id for n in batch]
            metadatas = [self._sanitize_metadata(n.metadata) for n in batch]

            embeddings = self._embedder.embed_texts(texts)

            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            indexed += len(batch)
            logger.debug(f"  Batch {i//batch_size + 1}: {len(batch)} chunk indicizzati")

        logger.info(f"Indicizzazione completata. Totale collection: {self._collection.count()}")
        return indexed

    # ── Ricerca ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Ricerca i chunk più rilevanti per una query.

        Args:
            query   : testo della query (o domanda requisito in fase inferenza)
            top_k   : numero di risultati da restituire
            filters : filtri ChromaDB (es. {"source": "policy_aml.pdf"})

        Returns:
            Lista di dict con: id, text, metadata, distance
        """
        query_embedding = self._embedder.embed_query(query)

        where = filters if filters else None

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Normalizzazione output
        output = []
        for i, doc_id in enumerate(results["ids"][0]):
            output.append({
                "id": doc_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],  # cosine → similarity
            })

        return output

    # ── Utilità ───────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Numero di chunk nella collection."""
        return self._collection.count()

    def reset_collection(self) -> None:
        """Elimina e ricrea la collection (utile per re-indicizzazione completa)."""
        logger.warning(f"Reset collection '{self.collection_name}'")
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """
        ChromaDB accetta solo stringhe, int, float, bool come valori nei metadati.
        Converte o rimuove i tipi non supportati.
        """
        sanitized = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            elif isinstance(v, list):
                # Converti liste in stringa JSON-like
                sanitized[k] = str(v)
            elif v is None:
                sanitized[k] = ""
            else:
                sanitized[k] = str(v)
        return sanitized


# ─────────────────────────────────────────────────────────────────────────────
# USAGE STANDALONE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from preprocessing.pdf_parser import PDFParser
    from preprocessing.chunker import DocumentChunker

    if len(sys.argv) < 2:
        print("Usage: python vector_store.py <path_to_pdf>")
        sys.exit(1)

    # Pipeline completa
    parser = PDFParser()
    docs = parser.parse(sys.argv[1])

    chunker = DocumentChunker()
    nodes = chunker.chunk(docs)

    store = VectorStore(collection_name=CHROMA_COLLECTION_POLICIES)
    n_indexed = store.index_nodes(nodes)
    print(f"\nChunk indicizzati: {n_indexed}")
    print(f"Totale in ChromaDB: {store.count()}")

    # Test ricerca
    test_query = "requisiti di sicurezza informatica"
    print(f"\nTest ricerca: '{test_query}'")
    results = store.search(test_query, top_k=3)
    for r in results:
        print(f"  Score={r['score']:.3f} | {r['metadata'].get('section_path', 'N/A')}")
        print(f"  Testo: {r['text'][:120]}...")
        print()