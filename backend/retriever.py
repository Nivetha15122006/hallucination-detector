import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self):
        # Define paths relative to the file location
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(backend_dir, "..", "data"))
        
        index_path = os.path.join(data_dir, "knowledge_base.index")
        chunks_path = os.path.join(data_dir, "chunks.pkl")
        
        print("Loading retriever models and index...")
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError(
                f"Knowledge base index files not found at {index_path} or {chunks_path}. "
                "Please run 'python backend/build_knowledge_base.py' first to generate them."
            )
            
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
            
        print(f"Retriever loaded successfully! {self.index.ntotal} chunks indexed.")

    def retrieve(self, question: str, top_k: int = 3):
        """Retrieve the top_k most relevant chunks from the FAISS index based on semantic distance."""
        # Encode the question using sentence-transformers
        query_vector = self.st_model.encode([question])
        query_vector = np.array(query_vector, dtype=np.float32)
        
        # Search the index
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append({
                'text': chunk['text'],
                'topic': chunk['topic'],
                'url': chunk.get('url', ''),
                # flat L2 search returns distances; lower is more similar.
                'score': float(distances[0][rank])
            })
            
        return results