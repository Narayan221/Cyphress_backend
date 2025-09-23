import os
import re
import uuid
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pdf_utils import extract_text_from_pdf, chunk_text
from vector_store import VectorStore
from mistral_client import ask_mistral
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv("MISTRAL_API_KEY")

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

    vector_store = VectorStore(API_KEY)
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

    # ----------------- Greeting and basic conversation check FIRST -----------------
    if q_lower in ['hi', 'hello', 'hey']:
        return {"answer": "Hi! How can I help you?", "memory": {}, "pdf_uploaded": False, "query_type": "greeting"}
    
    # Basic conversational responses
    conversational_queries = {
        'how are you': "I'm doing well, thank you! How can I assist you today?",
        'how are you?': "I'm doing well, thank you! How can I assist you today?",
        'how do you do': "I'm doing great! What can I help you with?",
        'thank you': "You're welcome! Is there anything else I can help you with?",
        'thanks': "You're welcome! Is there anything else I can help you with?",
        'bye': "Goodbye! Feel free to come back if you need any help.",
        'goodbye': "Goodbye! Feel free to come back if you need any help."
    }
    
    if q_lower in conversational_queries:
        return {"answer": conversational_queries[q_lower], "memory": memory, "pdf_uploaded": bool(vector_store), "query_type": "conversation"}

    # ----------------- Personal info sharing detection -----------------
    sharing_keywords = ['my ', 'i am ', "i'm ", 'i work ', 'i live ', 'i have ']
    is_query = query.endswith('?')
    
    if any(q_lower.startswith(kw) for kw in sharing_keywords) and not is_query:
        # Direct extraction with AI
        extract_prompt = f"""
Extract personal information. Return format: key:value (one per line)

"My name is John" -> name:John
"My age is 25" -> age:25
"My email is test@gmail.com" -> email:test@gmail.com
"I am 30 years old" -> age:30
"My phone is 123-456" -> phone:123-456

Message: {query}"""
        
        try:
            extraction = ask_mistral(extract_prompt, query)
            if extraction and ':' in extraction:
                lines = extraction.strip().split('\n')
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        memory[key] = value
                
                return {"answer": "Got it!", "memory": memory, "pdf_uploaded": bool(vector_store), "query_type": "personal"}
        except:
            pass
    
    # Identity questions - check FIRST regardless of memory
    identity_queries = ['do you know who i am', 'who am i', 'do you know me', 'know who i am']
    if any(q in q_lower for q in identity_queries):
        if memory and 'name' in memory:
            info_parts = [f"your name is {memory['name']}"]
            if 'age' in memory:
                info_parts.append(f"you are {memory['age']} years old")
            return {"answer": f"Yes, {', '.join(info_parts)}.", "memory": memory, "pdf_uploaded": bool(vector_store), "query_type": "personal"}
        else:
            return {"answer": "You haven't told me your name yet.", "memory": memory, "pdf_uploaded": bool(vector_store), "query_type": "personal"}
    
    # Check if querying stored personal info - PRIORITY CHECK
    if memory:
        # Direct personal info queries
        personal_queries = {
            'name': ['what is my name', 'my name', 'whats my name', "what's my name"],
            'age': ['what is my age', 'my age', 'whats my age', "what's my age", 'how old am i'],
            'email': ['what is my email', 'my email', 'whats my email', "what's my email"],
            'phone': ['what is my phone', 'my phone', 'whats my phone', "what's my phone"]
        }
        
        for key, queries in personal_queries.items():
            if key in memory and any(q in q_lower for q in queries):
                return {"answer": f"Your {key} is {memory[key]}.", "memory": memory, "pdf_uploaded": bool(vector_store), "query_type": "personal"}
        
        # Generic check for other stored keys
        for key in memory.keys():
            if f"my {key}" in q_lower or f"what is my {key}" in q_lower:
                return {"answer": f"Your {key} is {memory[key]}.", "memory": memory, "pdf_uploaded": bool(vector_store), "query_type": "personal"}
    
    # Check for PDF upload status queries
    upload_queries = ["uploaded", "upload", "pdf yet", "have i uploaded"]
    if any(term in q_lower for term in upload_queries):
        if vector_store is not None:
            return {"answer": "Yes, you have uploaded a PDF.", "memory": memory, "pdf_uploaded": True, "query_type": "general"}
        else:
            return {"answer": "No, you haven't uploaded a PDF yet.", "memory": memory, "pdf_uploaded": False, "query_type": "general"}
    
    # Check for PDF name queries
    pdf_name_queries = ["name of pdf", "pdf name", "name of the pdf", "what is the pdf name", "filename", "name in pdf"]
    if any(query in q_lower for query in pdf_name_queries):
        pdf_name = session_data.get("pdf_name")
        if pdf_name:
            return {"answer": f"The PDF name is {pdf_name}.", "memory": memory, "pdf_uploaded": bool(vector_store), "query_type": "general"}
        else:
            return {"answer": "No PDF has been uploaded yet.", "memory": memory, "pdf_uploaded": False, "query_type": "general"}

    


    # ----------------- Detect query type -----------------
    if any(word in q_lower for word in ["project", "work", "task", "report"]):
        query_type = "professional"
    elif any(word in q_lower for word in ["name", "age", "birthday", "email"]):
        query_type = "personal"
    else:
        query_type = "general"

    # ----------------- Check if asking about PDF content when no PDF exists -----------------
    pdf_content_keywords = [
        'pdf', 'document', 'file', 'inside', 'mentioned', 'name inside', 'experience', 'candidate',
        'author', 'writer', 'resume', 'cv', 'skills', 'qualification', 'education', 'work experience'
    ]
    if vector_store is None and any(keyword in q_lower for keyword in pdf_content_keywords):
        return {"answer": "No PDF has been uploaded yet. Please upload a PDF first.", "memory": memory, "pdf_uploaded": False, "query_type": "general"}
    
    # ----------------- PDF context -----------------
    if vector_store is not None and hasattr(vector_store, "get_relevant_chunks"):
        relevant_chunks = vector_store.get_relevant_chunks(query)
        context_text = "\n".join(relevant_chunks) if relevant_chunks else ""
    else:
        context_text = None

    # ----------------- Memory context -----------------
    memory_text = "\n".join([f"{k}: {v}" for k, v in memory.items()]) if memory else ""

    # ----------------- Construct smart prompt -----------------
    # Determine if question is about document content
    document_keywords = [
        'document', 'pdf', 'file', 'text', 'mentioned', 'experience', 'he', 'she', 'person', 'candidate',
        'author', 'writer', 'created', 'written', 'resume', 'cv', 'profile', 'skills', 'qualification',
        'education', 'work', 'job', 'company', 'position', 'role', 'responsibility', 'achievement'
    ]
    
    is_document_question = (
        context_text and (
            any(word in q_lower for word in document_keywords) or
            not any(word in q_lower for word in ['my', 'i am', 'i have', 'me'])
        )
    )
    
    # Check if we have relevant context to answer
    has_context = bool(memory_text) or is_document_question
    
    if not has_context:
        return {"answer": "I can only help with questions about your personal information or the uploaded document. Please share some personal details or upload a PDF first.", "memory": memory, "pdf_uploaded": bool(context_text), "query_type": "general"}
    
    prompt = f"""
You are Cyphress AI. Answer ONLY based on the provided context below. Do not use general knowledge.

RULES:
- If the answer is not in the provided context, say "I don't have that information in the available context."
- Only use the personal info or document content provided
- Be concise and direct
"""

    if memory_text:
        prompt += f"\nUser's Personal Info: {memory_text}\n"

    if is_document_question:
        prompt += f"\nDocument Content: {context_text}\n"

    prompt += f"\nQuestion: {query}\nAnswer:"

    # ----------------- Get AI Response -----------------
    try:
        response_text = ask_mistral(prompt, query)
        if not response_text:
            response_text = "I'm not sure how to help with that."
    except:
        response_text = "I'm not sure how to help with that."
    
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


# ------------------- Clear Chat -------------------
@app.post("/clear_chat/")
async def clear_chat(session_id: str = Form(...)):
    if session_id not in sessions:
        return {"error": "Invalid session"}

    # Only clear memory, keep PDF data
    sessions[session_id]["memory"] = {}

    return {"status": "Chat cleared"}