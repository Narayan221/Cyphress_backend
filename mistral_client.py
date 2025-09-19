import os
from mistralai import Mistral
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv("MISTRAL_API_KEY")  
if not API_KEY:
    raise ValueError("MISTRAL_API_KEY not found. Make sure .env is set correctly.")
client = Mistral(api_key=API_KEY)

def get_embedding(text: str):
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=text
    )
    return response.data[0].embedding

def ask_mistral(context, query):
    system_prompt = (
        "You are a helpful AI assistant. Your knowledge is limited to the uploaded PDF. "
        "If the answer is not in the PDF, say so politely. Use clear and concise language."
    )

    user_prompt = f"User asked: {query}\n\nRefer to this excerpt from the PDF:\n{context}\n\nAnswer the user query based only on this content."

    response = client.chat.complete(
        model="mistral-tiny", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    return response.choices[0].message.content
