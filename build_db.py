from sentence_transformers import SentenceTransformer
import chromadb
import re

DB_PATH = "/workspace/chroma_db"

model = SentenceTransformer('all-MiniLM-L6-v2')

db = chromadb.PersistentClient(path=DB_PATH)
collection = db.get_or_create_collection("laws")

with open("pakistan_rental_laws_consolidated.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split by double newlines or sections (### or =====)
chunks = re.split(r'\n\s*\n|\n===|###', text)

# Clean chunks
chunks = [c.strip() for c in chunks if len(c.strip()) > 100]

# Limit to avoid embedding issues (max 1000 chunks)
chunks = chunks[:1000]

for i, chunk in enumerate(chunks):
    emb = model.encode(chunk).tolist()
    collection.add(
        ids=[f"chunk_{i}"],
        embeddings=[emb],
        documents=[chunk],
        metadatas=[{"index": i, "length": len(chunk)}]
    )

print(f"Vector database built with {len(chunks)} chunks")