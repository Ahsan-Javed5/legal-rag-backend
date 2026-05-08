from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import re
import pypdf
from io import BytesIO
from vllm import LLM, SamplingParams
import os

app = FastAPI()

DB_PATH = "/workspace/chroma_db"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"  # Lightweight, ya "mistralai/Mistral-7B-Instruct-v0.3"

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load ChromaDB
db = chromadb.PersistentClient(path=DB_PATH)
collection = db.get_or_create_collection("laws")

# Load vLLM (only once at startup)
print(f"Loading vLLM model: {MODEL_NAME}...")
try:
    llm = LLM(model=MODEL_NAME, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=500)
    print("vLLM loaded successfully.")
except Exception as e:
    print(f"Warning: vLLM failed to load: {e}")
    llm = None

class Query(BaseModel):
    question: str

class Agreement(BaseModel):
    text: str

# ========== PROMPT TEMPLATE (Synonym-Aware) ==========
SYSTEM_PROMPT = """You are a Pakistani rental law expert. The user may use synonyms like:
- Tenant = lessee, renter, occupant, resident, leaseholder, lodger
- Landlord = lessor, owner, proprietor, house owner
- Rent = lease amount, hire charge, occupancy fee
- Eviction = ejectment, dispossession
- Agreement = contract, lease deed

Your task: Analyze the given clause against the retrieved Pakistani rental law.
Answer in this exact format:
- Violation: Yes/No
- Section: (if violation, mention act and section, e.g., "Punjab Act Section 6")
- Reason: (one sentence explanation)

If no violation: "Compliant"
"""

def call_vllm(prompt: str) -> str:
    """Call vLLM with prompt, return response"""
    if llm is None:
        return "Error: vLLM not loaded. Please check model availability."
    
    try:
        outputs = llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()
    except Exception as e:
        return f"vLLM error: {str(e)}"

def analyze_text(text: str):
    """Split agreement into clauses, retrieve law, analyze with vLLM"""
    clauses = re.split(r'[.\n]+', text)
    clauses = [c.strip() for c in clauses if len(c.strip()) > 30]
    results = []

    for i, clause in enumerate(clauses):
        # 1. Retrieve relevant law from ChromaDB
        emb = embedding_model.encode(clause).tolist()
        retrieved = collection.query(
            query_embeddings=[emb],
            n_results=1
        )
        law_text = retrieved["documents"][0][0] if retrieved["documents"] else "No specific law found in database."

        # 2. Build final prompt
        user_prompt = f"""Clause from rental agreement:
"{clause}"

Retrieved relevant Pakistani rental law:
"{law_text}"

Based on the above law, analyze the clause."""

        final_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

        # 3. Call vLLM
        analysis = call_vllm(final_prompt)

        results.append({
            "clause_num": i + 1,
            "clause_text": clause[:300],
            "retrieved_law_preview": law_text[:200],
            "analysis": analysis
        })

    return {
        "total_clauses": len(clauses),
        "results": results
    }

# ========== API ENDPOINTS ==========
@app.post("/ask")
def ask(query: Query):
    """Simple question-answering endpoint"""
    emb = embedding_model.encode(query.question).tolist()
    retrieved = collection.query(query_embeddings=[emb], n_results=1)
    law_text = retrieved["documents"][0][0] if retrieved["documents"] else "No match found"
    
    final_prompt = f"{SYSTEM_PROMPT}\n\nUser question: {query.question}\n\nRelevant law: {law_text}\n\nAnswer the question based on the law."
    
    answer = call_vllm(final_prompt)
    
    return {
        "question": query.question,
        "answer": answer,
        "retrieved_law": law_text[:500]
    }

@app.post("/analyze-agreement")
def analyze_agreement(agreement: Agreement):
    """Analyze full rental agreement text"""
    return analyze_text(agreement.text)

@app.post("/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    """Upload and analyze a PDF rental agreement"""
    contents = await file.read()
    pdf = pypdf.PdfReader(BytesIO(contents))
    
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    
    return analyze_text(text)

@app.get("/health")
def health_check():
    return {
        "status": "running",
        "vllm_loaded": llm is not None,
        "chroma_collection_count": collection.count(),
        "model": MODEL_NAME
    }