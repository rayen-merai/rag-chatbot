import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
col = chromadb.PersistentClient(path="chroma_db").get_collection("docs")

print("Chunks in database:", col.count())

while True:
    q = input("\nQuestion (empty to quit): ").strip()
    if not q:
        break
    res = col.query(query_embeddings=model.encode([q]).tolist(), n_results=4)
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        print(f"\n[{meta['source']} p.{meta['page']}] distance={dist:.2f}")
        print(doc[:300])