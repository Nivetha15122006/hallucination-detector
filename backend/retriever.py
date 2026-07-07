import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.chunks = None
        self._load()

    def _load(self):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        index_path = os.path.join(base_dir, 'data', 'knowledge_base.index')
        chunks_path = os.path.join(base_dir, 'data', 'chunks.pkl')

        if os.path.exists(index_path) and os.path.exists(chunks_path):
            self.index = faiss.read_index(index_path)
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            print(f"Retriever loaded! {self.index.ntotal} vectors")
        else:
            print("Warning: knowledge base files not found!")

    def retrieve(self, question: str, top_k: int = 3):
        if self.index is None:
            return []

        embedding = self.model.encode([question])
        distances, indices = self.index.search(
            np.array(embedding, dtype=np.float32),
            top_k
        )

        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'text': self.chunks[idx]['text'],
                'topic': self.chunks[idx]['topic'],
                'score': float(distances[0][i])
            })
        return results