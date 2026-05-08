from sentence_transformers import SentenceTransformer
import chromadb
import re

DB_PATH = "/workspace/chroma_db"

model = SentenceTransformer('all-MiniLM-L6-v2')

db = chromadb.PersistentClient(path=DB_PATH)

# DELETE OLD COLLECTION
try:
    db.delete_collection("laws")
except:
    pass

collection = db.create_collection("laws")

# =========================
# LOAD LAW FILE
# =========================
with open("rental_acts.txt", "r", encoding="utf-8") as f:
    text = f.read()

# =========================
# BETTER CHUNKING
# =========================
sections = re.split(
    r'(Section\s+\d+.*?:|SECTION\s+\d+.*?:)',
    text
)

chunks = []

current = ""

for part in sections:

    part = part.strip()

    if not part:
        continue

    if re.match(r'(Section|SECTION)\s+\d+', part):

        if current:
            chunks.append(current.strip())

        current = part

    else:
        current += "\n" + part

if current:
    chunks.append(current.strip())

# CLEAN
chunks = [
    c for c in chunks
    if len(c) > 80
]

print("Total chunks:", len(chunks))

# =========================
# STORE EMBEDDINGS
# =========================
for i, chunk in enumerate(chunks):

    emb = model.encode(chunk).tolist()

    collection.add(
        ids=[f"law_{i}"],
        embeddings=[emb],
        documents=[chunk],
        metadatas=[{
            "source": "pakistani_rental_law",
            "chunk_id": i
        }]
    )

print("Database built successfully.")