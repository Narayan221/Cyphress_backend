import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.model = SentenceTransformer(model_name)

    def build_index(self, chunks):
        self.chunks = chunks
        self.embeddings = self.model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

        dim = self.embeddings.shape[1] 
        self.index = faiss.IndexFlatIP(dim) 
        self.index.add(self.embeddings)

    def get_relevant_chunks(self, query, top_k=5):
        if not self.index:
            return []
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, top_k)
        # Retrieve corresponding chunks
        relevant_chunks = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        return relevant_chunks
