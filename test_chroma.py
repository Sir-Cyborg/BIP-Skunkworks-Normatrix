import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./chroma_db", settings=Settings(anonymized_telemetry=False))
col = client.get_collection("bank_policies")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

query = "gestione degli incidenti informatici"
embedding = model.encode(query).tolist()

results = col.query(
    query_embeddings=[embedding],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

for i, (doc, meta, dist) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
)):
    print(f"\n[{i+1}] Score={1-dist:.3f} | {meta.get('section_path','N/A')}\n\n\n")
    print(f"     {doc[:250]}...")