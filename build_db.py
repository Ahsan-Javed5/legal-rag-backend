from sentence_transformers import SentenceTransformer
import chromadb
import re

DB_PATH = "/workspace/chroma_db"

# =========================
# LOAD EMBEDDING MODEL
# =========================
model = SentenceTransformer('all-MiniLM-L6-v2')

# =========================
# LOAD / RESET CHROMADB
# =========================
db = chromadb.PersistentClient(path=DB_PATH)

try:
    db.delete_collection("laws")
except:
    pass

collection = db.create_collection("laws")

# =========================
# LOAD LAW FILE (simplified clauses)
# =========================
with open("rental_acts.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# =========================
# CHUNKING: each numbered clause becomes its own chunk
# =========================
chunks = []
current_chunk = ""

for line in lines:
    line = line.strip()
    if not line:
        continue                     # skip empty lines

    # A clause starts with a pattern like "5.1:" or "10.1:", etc.
    # We treat each such line as its own chunk.
    # This way every rule is independent and easy to retrieve.
    if re.match(r'^\d+\.\d+:', line):
        if current_chunk:
            chunks.append(current_chunk)
            current_chunk = ""
        current_chunk = line
    else:
        # In case a clause spans two lines (unlikely in your clean file)
        if current_chunk:
            current_chunk += " " + line
        else:
            current_chunk = line

# Add last chunk
if current_chunk:
    chunks.append(current_chunk)

# Remove duplicates (preserve order)
seen = set()
unique_chunks = []
for ch in chunks:
    if ch not in seen:
        seen.add(ch)
        unique_chunks.append(ch)

chunks = unique_chunks

# Debug info
print(f"Total chunks: {len(chunks)}")
if chunks:
    print("\nFirst 3 chunks:\n")
    for i, ch in enumerate(chunks[:3]):
        print(f"{i+1}. {ch[:150]}...\n")

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
            "source": "islamabad_rent_law_simplified",
            "chunk_id": i,
            "length": len(chunk)
        }]
    )

print(f"\nDatabase built successfully with {len(chunks)} chunks.")