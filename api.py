from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import re
import pypdf
from io import BytesIO

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')
db = chromadb.PersistentClient(path="./law_db")
collection = db.get_collection("laws")

class Query(BaseModel):
    question: str

class Agreement(BaseModel):
    text: str

def analyze_text(text: str):
    clauses = re.split(r'[.\n]+', text)
    clauses = [c.strip() for c in clauses if len(c.strip()) > 30]
    
    results = []
    for i, clause in enumerate(clauses):
        emb = model.encode(clause).tolist()
        retrieved = collection.query(query_embeddings=[emb], n_results=1)
        law = retrieved['documents'][0][0]
        results.append({
            "clause_num": i+1,
            "text": clause[:200],
            "law": law[:300]
        })
    return {"total_clauses": len(clauses), "results": results}

@app.post("/ask")
def ask(query: Query):
    emb = model.encode(query.question).tolist()
    results = collection.query(query_embeddings=[emb], n_results=1)
    law = results['documents'][0][0]
    return {"answer": law, "law": law}

@app.post("/analyze-agreement")
def analyze_agreement(agreement: Agreement):
    return analyze_text(agreement.text)

@app.post("/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    pdf = pypdf.PdfReader(BytesIO(contents))
    
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    
    return analyze_text(text)