import chromadb

# Connessione HTTP al server Docker
client = chromadb.HttpClient(host="localhost", port=8000)

print("Collection:", [c.name for c in client.list_collections()])

col = client.get_collection("bank_policies")
print(f"Totale chunk: {col.count()}")

results = col.get(limit=5, include=["documents", "metadatas"])

for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Sezione : {meta.get('section_path', 'N/A')}")
    print(f"Pagina  : {meta.get('page_number', 'N/A')}")
    print(f"Testo   : {doc[:300]}...")