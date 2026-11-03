import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/chunks.csv")
print("Loading model (first run downloads it)...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)
print("Embeddings shape:", embeddings.shape)       # (number of chunks, 384)

client = chromadb.PersistentClient(path="chroma_db")
try:
    client.delete_collection("docs")               # rebuild from scratch each run
except Exception:
    pass
col = client.create_collection("docs", metadata={"hnsw:space": "cosine"})

BATCH = 500
for i in range(0, len(df), BATCH):
    part = df.iloc[i:i + BATCH]
    col.add(
        ids=part["id"].tolist(),
        documents=part["text"].tolist(),
        embeddings=embeddings[i:i + BATCH].tolist(),
        metadatas=[{"source": s, "page": int(p)} for s, p in zip(part["source"], part["page"])],
    )

print("Chunks stored in ChromaDB:", col.count())