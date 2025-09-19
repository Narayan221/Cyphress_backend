import os
import re
import uuid
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from backend.pdf_utils import extract_text_from_pdf, chunk_text
from backend.vector_store import VectorStore
from mistral_client import ask_mistral

app = FastAPI()
sessions = {}

class AskRequest(BaseModel):
    query: str
    session_id: str 

# ------------------- CORS Middleware -------------------    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Create Session -------------------
@app.post("/create_session/")
async def create_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"vector_store": None, "memory": {}, "pdf_name": None}
    return {"session_id": session_id}

# ------------------- Upload PDF -------------------
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile, session_id: str = Form(...)):
    if session_id not in sessions:
        sessions[session_id] = {"vector_store": None, "memory": {}, "pdf_name": None}

    file_bytes = await file.read()
    text = extract_text_from_pdf(file_bytes)
    chunks = chunk_text(text)

    vector_store = VectorStore()
    vector_store.build_index(chunks)

    # Save in session
    sessions[session_id]["vector_store"] = vector_store
    sessions[session_id]["pdf_name"] = file.filename

    return {"status": "success", "chunks": len(chunks), "filename": file.filename}

@app.post("/ask/")
async def ask_question(request: AskRequest):
    session_id = request.session_id
    if session_id not in sessions:
        return {"error": "Invalid session. Please create a session first."}

    session_data = sessions[session_id]
    memory = session_data["memory"]
    vector_store = session_data["vector_store"]

    query = request.query.strip()
    q_lower = query.lower()

    # ----------------- Update memory -----------------
    match_name = re.search(r"my name is (\w+ \w+|\w+)", query, re.IGNORECASE)
    if match_name:
        memory["name"] = match_name.group(1)

    match_age = re.search(r"i am (\d+) years? old", query, re.IGNORECASE)
    if match_age:
        memory["age"] = int(match_age.group(1))

    # ----------------- Detect query type -----------------
    if any(word in q_lower for word in ["project", "work", "task", "report"]):
        query_type = "professional"
    elif any(word in q_lower for word in ["name", "age", "birthday", "email"]):
        query_type = "personal"
    else:
        query_type = "general"

    # ----------------- PDF context -----------------
    if vector_store and hasattr(vector_store, "get_relevant_chunks"):
        relevant_chunks = vector_store.get_relevant_chunks(query)
        context_text = "\n".join(relevant_chunks) if relevant_chunks else ""
    else:
        context_text = None

    # ----------------- Memory context -----------------
    memory_text = "\n".join([f"{k}: {v}" for k, v in memory.items()]) if memory else ""

    # ----------------- Construct smart prompt -----------------
    prompt = f"""
You are Cyphress AI, a professional, friendly, and concise assistant.
Your goal is to answer user queries as clearly and accurately as possible.

Instructions:
- Answer in 1-2 short sentences.
- Professional tone for work queries, friendly for personal, informative for general.
- Avoid filler words or unnecessary explanations.
- Use memory and PDF content if relevant.
- Do NOT mention the classification to the user.

"""

    if memory_text:
        prompt += f"\nMemory:\n{memory_text}\n"

    if context_text:
        prompt += f"\nRefer to PDF:\n{context_text}\n"

    prompt += f"\nQuestion: {query}"

    # ----------------- Get AI Response -----------------
    response_text = ask_mistral(prompt, query)
    return {
        "answer": response_text,
        "memory": memory,
        "pdf_uploaded": bool(context_text),
        "query_type": query_type
    }



# ------------------- Delete PDF -------------------
@app.post("/delete_pdf/")
async def delete_pdf(session_id: str = Form(...)):
    if session_id not in sessions:
        return {"error": "Invalid session"}

    sessions[session_id]["vector_store"] = None
    sessions[session_id]["pdf_name"] = None

    return {"status": "PDF deleted"}


@app.post("/validate_session/")
async def validate_session(data: dict):
    session_id = data.get("session_id")
    if session_id in sessions:
        return {"valid": True}
    return {"valid": False}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.app:app", host="0.0.0.0", port=port, reload=False)