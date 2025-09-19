import faiss
import numpy as np
from mistralai import Mistral

class VectorStore:
    def __init__(self, api_key):
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.client = Mistral(api_key=api_key)

    def build_index(self, chunks):
        self.chunks = chunks
        embeddings_response = self.client.embeddings.create(
            model="mistral-embed",
            inputs=chunks
        )
        self.embeddings = np.array([emb.embedding for emb in embeddings_response.data])
        
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def get_relevant_chunks(self, query, top_k=5):
        if not self.index:
            return []
        query_response = self.client.embeddings.create(
            model="mistral-embed",
            inputs=[query]
        )
        query_embedding = np.array([query_response.data[0].embedding])
        
        scores, indices = self.index.search(query_embedding, top_k)
        relevant_chunks = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        return relevant_chunks
