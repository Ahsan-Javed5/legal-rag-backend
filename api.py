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

# =========================
# LOAD EMBEDDING MODEL
# =========================
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# =========================
# LOAD CHROMADB
# =========================
db = chromadb.PersistentClient(path=DB_PATH)
collection = db.get_or_create_collection("laws")

# =========================
# LOAD VLLM
# =========================
print(f"Loading vLLM model: {MODEL_NAME}")

try:
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=4096
    )

    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.95,
        max_tokens=300
    )

    print("vLLM loaded successfully.")

except Exception as e:
    print("vLLM load failed:", e)
    llm = None


# =========================
# REQUEST MODELS
# =========================
class Query(BaseModel):
    question: str


class Agreement(BaseModel):
    text: str


# =========================
# IMPROVED SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """
You are an expert Pakistani rental law AI assistant.

The user may describe a rental clause in short form.

Your task:
1. Determine whether the clause/question violates Pakistani rental law.
2. Use ONLY the retrieved law context.
3. Do NOT invent laws or sections.
4. If the retrieved law is insufficient, say:
   "Insufficient legal context found."

Return STRICTLY in this format:

Violation: Yes/No
Section: <Act and section OR N/A>
Reason: <short explanation>

If compliant:
Violation: No
Section: N/A
Reason: Clause appears compliant with retrieved law.
"""


# =========================
# VLLM CALL
# =========================
def call_vllm(prompt: str):

    if llm is None:
        return "vLLM not loaded."

    try:
        outputs = llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

    except Exception as e:
        return str(e)


# =========================
# QUERY NORMALIZATION
# =========================
def normalize_query(q: str):

    q = q.strip()

    templates = [
        f"Is this legal under Pakistani rental law: {q}?",
        f"Does this violate Pakistani tenancy law: {q}?",
        f"Rental clause: {q}"
    ]

    return " ".join(templates)


# =========================
# RETRIEVE LAW
# =========================
def retrieve_laws(query, n_results=3):

    processed_query = normalize_query(query)

    emb = embedding_model.encode(processed_query).tolist()

    retrieved = collection.query(
        query_embeddings=[emb],
        n_results=n_results
    )

    docs = []

    if retrieved["documents"]:
        docs = retrieved["documents"][0]

    return docs


# =========================
# ANALYZE SINGLE QUERY
# =========================
def analyze_question(question: str):

    laws = retrieve_laws(question)

    combined_laws = "\n\n".join(laws)

    prompt = f"""
{SYSTEM_PROMPT}

USER QUESTION:
{question}

RETRIEVED PAKISTANI RENTAL LAW:
{combined_laws}

Analyze the user question against the law.
"""

    answer = call_vllm(prompt)

    return {
        "question": question,
        "retrieved_laws": laws,
        "answer": answer
    }


# =========================
# ANALYZE AGREEMENT
# =========================
def analyze_text(text: str):

    clauses = re.split(r'\n|\. ', text)

    clauses = [
        c.strip()
        for c in clauses
        if len(c.strip()) > 40
    ]

    results = []

    for i, clause in enumerate(clauses):

        laws = retrieve_laws(clause)

        combined_laws = "\n\n".join(laws)

        prompt = f"""
{SYSTEM_PROMPT}

CLAUSE:
{clause}

RETRIEVED LAW:
{combined_laws}

Analyze the clause.
"""

        analysis = call_vllm(prompt)

        results.append({
            "clause_number": i + 1,
            "clause": clause,
            "analysis": analysis,
            "retrieved_laws": laws
        })

    return {
        "total_clauses": len(clauses),
        "results": results
    }


# =========================
# ENDPOINTS
# =========================
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

    text = ""

    for page in pdf.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return analyze_text(text)


@app.get("/health")
def health():

    return {
        "status": "running",
        "model": MODEL_NAME,
        "laws_count": collection.count(),
        "vllm_loaded": llm is not None
    }