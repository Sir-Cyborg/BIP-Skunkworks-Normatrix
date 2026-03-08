"""
chunker.py - Chunking gerarchico con LlamaIndex

Strategia GERARCHICA:
  - Ogni Document (sezione) viene suddiviso in chunk a 3 livelli:
      large  (1024 token) → contesto ampio, utile per re-ranking
      medium  (512 token) → chunk di retrieval principale
      small   (128 token) → precisione fine-grained
  - I chunk figli mantengono riferimento al padre (parent_id)
  - I metadati della sezione vengono propagati a ogni chunk

Perché gerarchico per policy bancarie?
  Una domanda come "quali sono i requisiti di notifica per DORA art.17?"
  potrebbe matchare su un chunk piccolo, ma la risposta completa sta nel
  paragrafo intero. Con la gerarchia il retriever trova il chunk small/medium
  e il LLM riceve il contesto del chunk large padre → risposta più completa.
"""

from loguru import logger
from typing import Optional

from llama_index.core import Document
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SentenceWindowNodeParser,
    SentenceSplitter,
    get_leaf_nodes,
    get_root_nodes,
)
from llama_index.core.schema import BaseNode, NodeRelationship

from config import (
    CHUNK_STRATEGY,
    CHUNK_SIZES,
    CHUNK_OVERLAP,
)


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKER
# ─────────────────────────────────────────────────────────────────────────────

class DocumentChunker:
    """
    Suddivide i Document LlamaIndex in chunk pronti per l'embedding.

    Supporta tre strategie:
      - hierarchical    : chunk multi-livello (raccomandato)
      - sentence_window : chunk con finestra di frasi adiacenti come contesto
      - fixed           : chunk a dimensione fissa (baseline)
    """

    def __init__(
        self,
        strategy: str = CHUNK_STRATEGY,
        chunk_sizes: dict = CHUNK_SIZES,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.strategy = strategy
        self.chunk_sizes = chunk_sizes
        self.chunk_overlap = chunk_overlap
        self._parser = self._build_parser()

    # ── Factory ───────────────────────────────────────────────────────────────

    def _build_parser(self):
        if self.strategy == "hierarchical":
            return HierarchicalNodeParser.from_defaults(
                chunk_sizes=list(self.chunk_sizes.values()),  # [1024, 512, 128]
                chunk_overlap=self.chunk_overlap,
            )
        elif self.strategy == "sentence_window":
            return SentenceWindowNodeParser.from_defaults(
                window_size=3,           # frasi adiacenti come contesto
                window_metadata_key="window",
                original_text_metadata_key="original_text",
            )
        elif self.strategy == "fixed":
            return SentenceSplitter(
                chunk_size=self.chunk_sizes["medium"],
                chunk_overlap=self.chunk_overlap,
            )
        else:
            raise ValueError(f"Strategia sconosciuta: {self.strategy}")

    # ── Chunking ──────────────────────────────────────────────────────────────

    def chunk(self, documents: list[Document]) -> list[BaseNode]:
        """
        Suddivide una lista di Document in nodi/chunk.

        Args:
            documents: output del PDFParser

        Returns:
            Lista di BaseNode con testo, metadati e relazioni gerarchiche.
            Per la strategia 'hierarchical' include tutti i livelli.
            Per le altre, solo i leaf node.
        """
        if not documents:
            logger.warning("Nessun documento da chunkare.")
            return []

        logger.info(
            f"Chunking {len(documents)} document(s) | strategia={self.strategy}"
        )

        all_nodes = self._parser.get_nodes_from_documents(documents)
        logger.info(f"  Nodi totali generati: {len(all_nodes)}")

        # Propaga i metadati del documento padre a ogni nodo figlio
        all_nodes = self._propagate_metadata(all_nodes)

        if self.strategy == "hierarchical":
            leaf_nodes = get_leaf_nodes(all_nodes)
            logger.info(
                f"  Leaf node (per retrieval): {len(leaf_nodes)} "
                f"| Tutti i livelli: {len(all_nodes)}"
            )

        return all_nodes

    def get_leaf_nodes(self, nodes: list[BaseNode]) -> list[BaseNode]:
        """Restituisce solo i leaf node (chunk più piccoli, usati per retrieval)."""
        return get_leaf_nodes(nodes)

    def get_root_nodes(self, nodes: list[BaseNode]) -> list[BaseNode]:
        """Restituisce solo i root node (chunk più grandi, usati per contesto)."""
        return get_root_nodes(nodes)

    # ── Metadata propagation ──────────────────────────────────────────────────

    def _propagate_metadata(self, nodes: list[BaseNode]) -> list[BaseNode]:
        """
        Assicura che i metadati della sezione originale siano presenti
        in tutti i chunk figli (es. source, section_path, page_number).

        LlamaIndex propaga i metadata del Document ai nodi automaticamente,
        ma qui aggiungiamo anche informazioni sul livello del chunk.
        """
        for node in nodes:
            # Livello nella gerarchia
            node.metadata["chunk_level"] = self._detect_chunk_level(node)

            # Escludiamo dai metadati inline (quelli che LlamaIndex prepende al testo
            # durante il chunking) tutti i campi verbose o lunghi.
            # Questo evita il ValueError "Metadata length > chunk size".
            # I metadati restano comunque disponibili su node.metadata per ChromaDB.
            node.excluded_embed_metadata_keys = [
                "element_types",
                "char_count",
                "chunk_level",
                "section_path",    # può essere lungo, lo escludiamo dall'inline
                "section_title",   # idem
            ]
            node.excluded_llm_metadata_keys = [
                "element_types",
                "char_count",
                "chunk_level",
                "section_path",
                "section_title",
            ]

        return nodes

    def _detect_chunk_level(self, node: BaseNode) -> str:
        """
        Determina il livello del chunk in base alla sua dimensione.
        Utile come metadato per filtrare in fase di retrieval.
        """
        text_len = len(node.text.split())  # parole approssimative

        # Soglie approssimative (i token non sono esattamente parole)
        large_thresh = self.chunk_sizes["large"] * 0.6
        medium_thresh = self.chunk_sizes["medium"] * 0.6

        if text_len >= large_thresh:
            return "large"
        elif text_len >= medium_thresh:
            return "medium"
        else:
            return "small"

    # ── Diagnostica ───────────────────────────────────────────────────────────

    def print_stats(self, nodes: list[BaseNode]) -> None:
        """Stampa statistiche sui chunk generati."""
        if not nodes:
            print("Nessun nodo.")
            return

        from collections import Counter
        levels = Counter(n.metadata.get("chunk_level", "?") for n in nodes)
        sources = Counter(n.metadata.get("source", "?") for n in nodes)

        text_lens = [len(n.text) for n in nodes]
        avg_len = sum(text_lens) / len(text_lens)

        print(f"\n{'='*50}")
        print(f"STATISTICHE CHUNKING")
        print(f"{'='*50}")
        print(f"Totale nodi     : {len(nodes)}")
        print(f"Livelli         : {dict(levels)}")
        print(f"Sorgenti        : {dict(sources)}")
        print(f"Lunghezza media : {avg_len:.0f} caratteri")
        print(f"Min / Max       : {min(text_lens)} / {max(text_lens)} caratteri")
        print(f"{'='*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# USAGE STANDALONE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pdf_parser import PDFParser
    import sys

    if len(sys.argv) < 2:
        print("Usage: python chunker.py <path_to_pdf>")
        sys.exit(1)

    # Parsing
    parser = PDFParser()
    docs = parser.parse(sys.argv[1])

    # Chunking
    chunker = DocumentChunker(strategy="hierarchical")
    nodes = chunker.chunk(docs)
    chunker.print_stats(nodes)

    # Mostra qualche leaf node
    leaf_nodes = chunker.get_leaf_nodes(nodes)
    print(f"\nPrimi 3 leaf node:\n")
    for node in leaf_nodes[:3]:
        print(f"  ID           : {node.node_id[:8]}...")
        print(f"  Livello      : {node.metadata.get('chunk_level')}")
        print(f"  Sezione      : {node.metadata.get('section_path', 'N/A')}")
        print(f"  Testo (150c) : {node.text[:150]}...")
        print()