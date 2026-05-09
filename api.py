from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from io import BytesIO
from vllm import LLM, SamplingParams
import chromadb
import pypdf
import re

app = FastAPI()

DB_PATH = "/workspace/chroma_db"
COLLECTION_NAME = "laws"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

db = chromadb.PersistentClient(path=DB_PATH)
collection = db.get_collection(COLLECTION_NAME)

print(f"Loading model: {MODEL_NAME}")

try:
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=4096
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.2,
        max_tokens=350,
        stop=[
            "\n\nUser:",
            "\n\nQuestion:",
            "\n\nClause:"
        ]
    )

    print("vLLM loaded.")

except Exception as e:
    print("vLLM load failed:", e)
    llm = None


class Query(BaseModel):
    question: str


class Agreement(BaseModel):
    text: str


SYSTEM_PROMPT = """
You are an expert Pakistani rental law assistant.

Rules:
- Use ONLY the retrieved law clauses.
- Never repeat the same sentence.
- Never generate duplicate paragraphs.
- Be concise and factual.
- Mention exact clause numbers when relevant.
- If context is insufficient, clearly say so.

Response format:

Violation: Yes/No/Unknown

Relevant Clauses:
- clause number

Explanation:
Short precis explanation in plain English.
""".strip()


def clean_text(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_unique_lines(text: str):
    lines = text.splitlines()

    seen = set()
    final_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line.lower() not in seen:
            seen.add(line.lower())
            final_lines.append(line)

    return "\n".join(final_lines)


def call_vllm(prompt: str):
    if llm is None:
        return "vLLM not loaded."

    try:
        outputs = llm.generate([prompt], sampling_params)

        text = outputs[0].outputs[0].text.strip()

        text = extract_unique_lines(text)

        return text

    except Exception as e:
        return str(e)


def retrieve_laws(query: str, n_results=5):
    query = clean_text(query)

    embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    unique_docs = []
    seen = set()

    for doc, meta in zip(docs, metas):
        doc = clean_text(doc)

        if doc not in seen:
            seen.add(doc)

            unique_docs.append({
                "clause": meta.get("clause", "unknown"),
                "text": doc
            })

    return unique_docs


def build_context(laws):
    context_parts = []

    for law in laws:
        context_parts.append(
            f"Clause {law['clause']}: {law['text']}"
        )

    return "\n\n".join(context_parts)


def generate_response(user_input: str):
    laws = retrieve_laws(user_input)

    context = build_context(laws)

    prompt = f"""
{SYSTEM_PROMPT}

User Input:
{user_input}

Retrieved Law Clauses:
{context}

Generate only one final answer.
Do not repeat anything.
""".strip()

    answer = call_vllm(prompt)

    return {
        "query": user_input,
        "retrieved_laws": laws,
        "answer": answer
    }


def split_agreement(text: str):
    pattern = r"(?:\d+\.\s)(.*?)(?=\n\d+\.|\Z)"

    matches = re.findall(
        pattern,
        text,
        re.DOTALL
    )

    cleaned = []

    for clause in matches:
        clause = clean_text(clause)

        if len(clause) >= 20:
            cleaned.append(clause)

    return cleaned


def analyze_agreement_text(text: str):
    clauses = split_agreement(text)

    results = []

    for idx, clause in enumerate(clauses):
        response = generate_response(clause)

        results.append({
            "clause_number": idx + 1,
            "clause": clause,
            "analysis": response["answer"],
            "retrieved_laws": response["retrieved_laws"]
        })

    return {
        "total_clauses": len(results),
        "results": results
    }


@app.post("/ask")
def ask(query: Query):
    return generate_response(query.question)


@app.post("/analyze-agreement")
def analyze_agreement(agreement: Agreement):
    return analyze_agreement_text(agreement.text)


@app.post("/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    contents = await file.read()

    pdf = pypdf.PdfReader(BytesIO(contents))

    text = "\n".join(
        [page.extract_text() or "" for page in pdf.pages]
    )

    return analyze_agreement_text(text)


@app.get("/health")
def health():
    return {
        "status": "running",
        "model": MODEL_NAME,
        "laws_count": collection.count(),
        "vllm_loaded": llm is not None
    }