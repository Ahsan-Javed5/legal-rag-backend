from sentence_transformers import SentenceTransformer
import chromadb
import re

DB_PATH = "/workspace/chroma_db"

# =========================
# LOAD EMBEDDING MODEL
# =========================
model = SentenceTransformer('all-MiniLM-L6-v2')

# =========================
# LOAD CHROMADB
# =========================
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
# CHUNKING
# =========================
# Split by big section separators
raw_chunks = re.split(
    r'={10,}|###\s+SECTION',
    text
)

chunks = []

for chunk in raw_chunks:

    chunk = chunk.strip()

    if not chunk:
        continue

    # Remove very small chunks
    if len(chunk) < 100:
        continue

    # If chunk too large, split further
    if len(chunk) > 2000:

        sub_chunks = re.split(r'\n\s*\n+', chunk)

        for sub in sub_chunks:

            sub = sub.strip()

            if 100 < len(sub) < 2000:
                chunks.append(sub)

    else:
        chunks.append(chunk)

# Remove duplicates
chunks = list(dict.fromkeys(chunks))

# DEBUG
print("Total chunks:", len(chunks))
print("\nFIRST CHUNK PREVIEW:\n")
print(chunks[0][:500])

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
            "chunk_id": i,
            "length": len(chunk)
        }]
    )

print(f"\nDatabase built successfully with {len(chunks)} chunks.")