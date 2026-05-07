from sentence_transformers import SentenceTransformer
import chromadb
import re
import os

# Absolute persistent path (IMPORTANT)
DB_PATH = "/workspace/chroma_db"

model = SentenceTransformer('all-MiniLM-L6-v2')

db = chromadb.PersistentClient(path=DB_PATH)
collection = db.get_or_create_collection("laws")

with open("rent_act_2009.txt", "r", encoding="utf-8") as f:
    text = f.read()

sections = re.split(r'(Section \d+:)', text)

chunks = []
for i in range(1, len(sections), 2):
    if i + 1 < len(sections):
        chunks.append(sections[i] + sections[i + 1])

# clear old data (important for rebuilds)
collection.delete(where={})

for i, chunk in enumerate(chunks):
    emb = model.encode(chunk).tolist()
    collection.add(
        ids=[f"sec_{i}"],
        embeddings=[emb],
        documents=[chunk]
    )

print(f"Vector database built with {len(chunks)} sections at {DB_PATH}")