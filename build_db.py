from sentence_transformers import SentenceTransformer
import chromadb
import re

DB_PATH = "/workspace/chroma_db"
COLLECTION_NAME = "laws"

model = SentenceTransformer("all-MiniLM-L6-v2")

db = chromadb.PersistentClient(path=DB_PATH)

try:
    db.delete_collection(COLLECTION_NAME)
except:
    pass

collection = db.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

with open("rental_acts.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

pattern = r"(\d+\.\d+:\s.*?)(?=\n\d+\.\d+:|\Z)"
matches = re.findall(pattern, raw_text, re.DOTALL)

chunks = []
seen = set()

for match in matches:
    chunk = " ".join(match.split())

    if len(chunk) < 25:
        continue

    if chunk not in seen:
        seen.add(chunk)
        chunks.append(chunk)

print(f"Total chunks: {len(chunks)}")

embeddings = model.encode(
    chunks,
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=True
).tolist()

ids = [f"law_{i}" for i in range(len(chunks))]

metadatas = []

for i, chunk in enumerate(chunks):
    clause_match = re.match(r"^(\d+\.\d+):", chunk)

    metadatas.append({
        "clause": clause_match.group(1) if clause_match else "unknown",
        "source": "islamabad_rental_law",
        "chunk_id": i
    })

collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=chunks,
    metadatas=metadatas
)

print("Database built successfully.")