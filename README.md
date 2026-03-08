# RAG Preprocessing Pipeline — Policy & Procedure Bancarie

Pipeline di preprocessing per documenti aziendali (policy, procedure, regolamenti)
con parsing strutturato, chunking gerarchico e storage su ChromaDB.

---

## Architettura

```
PDF(s)
  │
  ▼
PDFParser (Unstructured.io)
  │  - Riconosce titoli, paragrafi, bullet point, tabelle
  │  - Raggruppa elementi per sezione logica
  │  - Arricchisce ogni sezione con metadati (source, section_path, page, ...)
  │
  ▼
DocumentChunker (LlamaIndex - HierarchicalNodeParser)
  │  - Livello large  (1024 token): sezione intera → contesto ampio
  │  - Livello medium  (512 token): paragrafo      → chunk di retrieval
  │  - Livello small   (128 token): frase          → precisione fine
  │
  ▼
VectorStore (ChromaDB + sentence-transformers multilingue)
     - Embedding: paraphrase-multilingual-mpnet-base-v2 (768 dim)
     - Distanza: coseno
     - Upsert idempotente (rieseguibile senza duplicati)
     - Collection separata per policies vs regulations
```

---

## Setup

```bash
# 1. Clona/copia il progetto
cd rag_preprocessing

# 2. Crea ambiente virtuale
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Installa dipendenze
pip install -r requirements.txt

# 4. (Solo per parsing hi_res con OCR — opzionale)
pip install "unstructured[all-docs]"
```

---

## Struttura del progetto

```
rag_preprocessing/
├── config.py          # Tutti i parametri configurabili
├── pdf_parser.py      # Parsing PDF con Unstructured.io
├── chunker.py         # Chunking gerarchico con LlamaIndex
├── vector_store.py    # Embedding + ChromaDB
├── pipeline.py        # Orchestratore CLI
├── requirements.txt
├── data/
│   └── raw/           # Metti qui i tuoi PDF
└── chroma_db/         # Generato automaticamente
```

---

## Utilizzo

### Come script CLI

```bash
# Indicizza tutti i PDF in data/raw/ nella collection "bank_policies"
python pipeline.py --input ./data/raw --collection bank_policies

# Indicizza un singolo PDF con reset della collection
python pipeline.py --input ./data/raw/policy_aml.pdf --reset

# Indicizza e poi esegui una query di test
python pipeline.py --input ./data/raw \
    --query "requisiti di notifica degli incidenti"

# Indicizza un regolamento (DORA, ecc.) nella collection dedicata
python pipeline.py \
    --input ./data/raw/DORA.pdf \
    --collection regulations
```

### Come modulo Python

```python
from pipeline import PreprocessingPipeline

# Indicizza il corpus di policy
pipeline = PreprocessingPipeline(collection_name="bank_policies")
pipeline.run("./data/raw")

# Ricerca manuale (utile per debug)
results = pipeline.search("continuità operativa", top_k=5)
for r in results:
    print(r["score"], r["metadata"]["section_path"])
    print(r["text"][:200])
```

### Test dei singoli moduli

```bash
# Solo parsing
python pdf_parser.py ./data/raw/policy_aml.pdf

# Parsing + chunking
python chunker.py ./data/raw/policy_aml.pdf

# Pipeline completa su un PDF
python vector_store.py ./data/raw/policy_aml.pdf
```

---

## Parametri configurabili (config.py)

| Parametro | Default | Descrizione |
|---|---|---|
| `EMBEDDING_MODEL` | `paraphrase-multilingual-mpnet-base-v2` | Modello embedding |
| `CHUNK_STRATEGY` | `hierarchical` | Strategia chunking |
| `CHUNK_SIZES` | `{large:1024, medium:512, small:128}` | Token per livello |
| `CHUNK_OVERLAP` | `64` | Overlap tra chunk |
| `UNSTRUCTURED_STRATEGY` | `auto` | Strategia parsing PDF |
| `MIN_TEXT_LENGTH` | `30` | Caratteri minimi per elemento |

---

## Metadati per chunk (disponibili in ChromaDB)

Ogni chunk indicizzato porta con sé:

| Metadato | Esempio | Utilizzo |
|---|---|---|
| `source` | `policy_aml.pdf` | Filtrare per documento |
| `section_path` | `Cap.3 > Sez.3.2 > Controlli` | Contestualizzare la risposta |
| `section_title` | `3.2 Controlli Antiriciclaggio` | Titolo della sezione |
| `page_number` | `12` | Riferimento pagina |
| `chunk_level` | `small` / `medium` / `large` | Filtrare per granularità |
| `has_table` | `True` | Sapere se il chunk contiene tabelle |
| `has_list` | `True` | Sapere se contiene bullet point |

---

## Note sul parsing PDF

Unstructured.io supporta tre strategie:

- **`fast`**: solo estrazione testo, nessuna OCR. Veloce, ma perde struttura complessa.
- **`hi_res`**: layout detection + OCR (richiede `detectron2`). Lento, ma molto accurato per PDF scansionati o con layout complesso.
- **`auto`** (default): sceglie automaticamente in base alle caratteristiche del PDF.

Per documenti bancari nativi digitali (non scansionati), `auto` o `fast` sono sufficienti.
Per documenti scansionati, usa `hi_res`.

---

## Prossimi step (fase di inferenza)

1. **Caricamento regolamento** (es. DORA): usa la stessa pipeline con `collection=regulations`
2. **Generazione domande requisito**: LLM converte ogni chunk del regolamento in domanda
3. **Retrieval**: ogni domanda interroga la collection `bank_policies`
4. **Valutazione**: LLM valuta se le evidenze recuperate soddisfano il requisito
