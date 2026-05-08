from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import re
import pypdf
from io import BytesIO
from vllm import LLM, SamplingParams

app = FastAPI()

DB_PATH = "/workspace/chroma_db"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
db = chromadb.PersistentClient(path=DB_PATH)
collection = db.get_or_create_collection("laws")

print(f"Loading vLLM model: {MODEL_NAME}")
try:
    llm = LLM(model=MODEL_NAME, trust_remote_code=True, gpu_memory_utilization=0.85, max_model_len=4096)
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=500)
    print("vLLM loaded.")
except Exception as e:
    print("vLLM load failed:", e)
    llm = None

class Query(BaseModel):
    question: str

class Agreement(BaseModel):
    text: str

SYSTEM_PROMPT = """You are a Pakistani rental law expert. Use ONLY the retrieved law context (numbered clauses like "5.1:", "10.4:" etc.). Do not invent laws.

If the user's clause or question violates any retrieved clause, output:
Violation: Yes
Affected Clauses: list the clause numbers (e.g., 5.1, 10.2)
Reason: one sentence explaining the violation.

If no violation:
Violation: No
Affected Clauses: N/A
Reason: The clause appears compliant.

If the retrieved law is insufficient:
Violation: Unknown
Affected Clauses: N/A
Reason: Insufficient legal context found.

Keep answers short and factual."""

def call_vllm(prompt: str):
    if llm is None:
        return "vLLM not loaded."
    try:
        outputs = llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()
    except Exception as e:
        return str(e)

def retrieve_laws(query, n_results=3):
    emb = embedding_model.encode(query).tolist()
    retrieved = collection.query(query_embeddings=[emb], n_results=n_results)
    docs = retrieved["documents"][0] if retrieved["documents"] else []
    return docs

def analyze_question(question: str):
    laws = retrieve_laws(question)
    combined = "\n\n".join(laws)
    prompt = f"""{SYSTEM_PROMPT}

User question: {question}

Retrieved law clauses:
{combined}

Answer strictly in the required format."""
    answer = call_vllm(prompt)
    return {"question": question, "retrieved_laws": laws, "answer": answer}

def analyze_text(text: str):
    clauses = [c.strip() for c in re.split(r'\n|\. ', text) if len(c.strip()) > 40]
    results = []
    for i, clause in enumerate(clauses):
        laws = retrieve_laws(clause)
        combined = "\n\n".join(laws)
        prompt = f"""{SYSTEM_PROMPT}

Clause: {clause}

Retrieved law clauses:
{combined}

Answer strictly in the required format."""
        analysis = call_vllm(prompt)
        results.append({"clause_number": i+1, "clause": clause, "analysis": analysis, "retrieved_laws": laws})
    return {"total_clauses": len(clauses), "results": results}

@app.post("/ask")
def ask(query: Query):
    return analyze_question(query.question)

@app.post("/analyze-agreement")
def analyze_agreement(agreement: Agreement):
    return analyze_text(agreement.text)

@app.post("/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    pdf = pypdf.PdfReader(BytesIO(contents))
    text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    return analyze_text(text)

@app.get("/health")
def health():
    return {"status": "running", "model": MODEL_NAME, "laws_count": collection.count(), "vllm_loaded": llm is not None}