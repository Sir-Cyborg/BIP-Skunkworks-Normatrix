"""
pipeline.py - Orchestratore della pipeline di preprocessing

Flusso completo:
    PDF(s) → PDFParser → DocumentChunker → VectorStore (ChromaDB)

Può essere usato:
  - Come script CLI: python pipeline.py --input ./data/raw --collection bank_policies
  - Come modulo importabile nella tua applicazione
"""

import argparse
import time
from pathlib import Path
from loguru import logger

from pdf_parser import PDFParser
from chunker import DocumentChunker
from vector_store import VectorStore
from config import (
    DATA_DIR,
    CHROMA_COLLECTION_POLICIES,
    CHROMA_COLLECTION_REGULATIONS,
    CHUNK_STRATEGY,
)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class PreprocessingPipeline:
    """
    Orchestratore della pipeline di preprocessing RAG.

    Esempio d'uso:
        pipeline = PreprocessingPipeline(collection_name="bank_policies")
        pipeline.run(input_path="./data/raw/policy_aml.pdf")
    """

    def __init__(
        self,
        collection_name: str = CHROMA_COLLECTION_POLICIES,
        chunk_strategy: str = CHUNK_STRATEGY,
        reset_collection: bool = False,
    ):
        logger.info("Inizializzazione pipeline di preprocessing...")

        self.parser = PDFParser()
        self.chunker = DocumentChunker(strategy=chunk_strategy)
        self.store = VectorStore(collection_name=collection_name)

        if reset_collection:
            logger.warning("Reset della collection richiesto.")
            self.store.reset_collection()

        logger.info(f"Pipeline pronta | collection='{collection_name}'")

    # ── Esecuzione ────────────────────────────────────────────────────────────

    def run(self, input_path: str | Path) -> dict:
        """
        Esegue la pipeline completa su un singolo PDF o una directory.

        Args:
            input_path: percorso a un PDF o a una directory di PDF

        Returns:
            Dizionario con statistiche dell'esecuzione
        """
        input_path = Path(input_path)
        start = time.time()

        # ── Step 1: Parsing ──────────────────────────────────────────────────
        logger.info(f"\n{'─'*50}")
        logger.info(f"STEP 1 | PARSING: {input_path.name}")
        logger.info(f"{'─'*50}")

        if input_path.is_dir():
            documents = self.parser.parse_directory(input_path)
        elif input_path.suffix.lower() == ".pdf":
            documents = self.parser.parse(input_path)
        else:
            raise ValueError(f"Input non supportato: {input_path}")

        if not documents:
            logger.error("Nessun documento estratto. Pipeline interrotta.")
            return {"status": "error", "reason": "no_documents"}

        logger.info(f"  ✓ Document estratti: {len(documents)}")

        # ── Step 2: Chunking ─────────────────────────────────────────────────
        logger.info(f"\n{'─'*50}")
        logger.info(f"STEP 2 | CHUNKING (strategia={self.chunker.strategy})")
        logger.info(f"{'─'*50}")

        nodes = self.chunker.chunk(documents)
        self.chunker.print_stats(nodes)

        # ── Step 3: Indicizzazione ───────────────────────────────────────────
        logger.info(f"\n{'─'*50}")
        logger.info(f"STEP 3 | INDICIZZAZIONE → ChromaDB")
        logger.info(f"{'─'*50}")

        n_indexed = self.store.index_nodes(nodes, only_leaf=True)

        elapsed = time.time() - start

        stats = {
            "status": "ok",
            "input": str(input_path),
            "documents_extracted": len(documents),
            "nodes_total": len(nodes),
            "nodes_indexed": n_indexed,
            "collection_total": self.store.count(),
            "elapsed_seconds": round(elapsed, 2),
        }

        logger.info(f"\n{'='*50}")
        logger.info(f"PIPELINE COMPLETATA in {elapsed:.1f}s")
        logger.info(f"  Document estratti : {stats['documents_extracted']}")
        logger.info(f"  Nodi totali       : {stats['nodes_total']}")
        logger.info(f"  Chunk indicizzati : {stats['nodes_indexed']}")
        logger.info(f"  Totale ChromaDB   : {stats['collection_total']}")
        logger.info(f"{'='*50}\n")

        return stats

    def search(self, query: str, top_k: int = 5, filters: dict = None) -> list[dict]:
        """Ricerca diretta sul VectorStore (utile per test rapidi)."""
        return self.store.search(query, top_k=top_k, filters=filters)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline di preprocessing RAG per documenti bancari"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(DATA_DIR),
        help="Percorso a un PDF o a una directory di PDF",
    )
    parser.add_argument(
        "--collection", "-c",
        type=str,
        default=CHROMA_COLLECTION_POLICIES,
        choices=[CHROMA_COLLECTION_POLICIES, CHROMA_COLLECTION_REGULATIONS],
        help="Nome della collection ChromaDB",
    )
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        default=CHUNK_STRATEGY,
        choices=["hierarchical", "sentence_window", "fixed"],
        help="Strategia di chunking",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Resetta la collection prima di indicizzare",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Query di test dopo l'indicizzazione",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pipeline = PreprocessingPipeline(
        collection_name=args.collection,
        chunk_strategy=args.strategy,
        reset_collection=args.reset,
    )

    stats = pipeline.run(args.input)

    if args.query and stats["status"] == "ok":
        print(f"\nRicerca di test: '{args.query}'\n")
        results = pipeline.search(args.query, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] Score={r['score']:.3f} | {r['metadata'].get('section_path', 'N/A')}")
            print(f"       {r['text'][:200]}...\n")
