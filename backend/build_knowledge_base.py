import os
# Configure HuggingFace cache directory to a local path to avoid permission errors
os.environ["HF_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hf_cache"))
os.environ["TRANSFORMERS_CACHE"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hf_cache"))

import requests
import time
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Try importing langchain text splitter, fallback to simple manual splitting if not installed
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    USE_LANGCHAIN = True
except ImportError:
    USE_LANGCHAIN = False

# List of 100+ popular Wikipedia topics
TOPICS = list(set([
    # Historical Figures
    "Albert Einstein", "Alexander Graham Bell", "Isaac Newton", "Marie Curie", "Mahatma Gandhi",
    "Nelson Mandela", "Napoleon", "Julius Caesar", "Cleopatra", "Leonardo da Vinci",
    "Nikola Tesla", "Stephen Hawking", "Charles Darwin", "Alexander the Great", "Abraham Lincoln",
    "Winston Churchill", "Queen Victoria", "George Washington", "Martin Luther King Jr.", "Galileo Galilei",
    "Joan of Arc", "Genghis Khan", "Gautama Buddha", "Socrates", "Aristotle", "Plato",
    
    # Historical Events & Ships
    "World War II", "World War I", "Titanic", "Apollo 11", "American Civil War", 
    "French Revolution", "Russian Revolution", "Industrial Revolution", "Magna Carta", "Renaissance",
    
    # Science & Astronomy
    "DNA", "Human brain", "Solar System", "Climate change", "Global warming", "Photosynthesis", 
    "Quantum mechanics", "Theory of relativity", "Plate tectonics", "Black hole", "Biodiversity", 
    "Renewable energy", "Evolution", "Mitochondria", "Milky Way", "Mars", "Moon", 
    "Sun", "Jupiter", "Saturn", "Supernova", "Big Bang", "Dark matter", "Dark energy", 
    "Higgs boson", "Large Hadron Collider", "Atom", "Molecule", "Water", "Carbon dioxide", 
    "Oxygen", "Nitrogen", "Hydrogen", "Helium", "Periodic table", "Chemical element",
    
    # Tech & Computing
    "Artificial intelligence", "Python (programming language)", "Internet", "World Wide Web", 
    "Cryptography", "Blockchain", "Machine learning", "Deep learning", "Quantum computing", 
    "Cloud computing", "Virtual reality", "Augmented reality", "3D printing", "Space exploration",
    "SpaceX", "NASA", "European Space Agency", "Hubble Space Telescope", "James Webb Space Telescope",
    "International Space Station", "Voyager 1", "Computer memory", "Operating system", "Linux", 
    "Microsoft Windows", "macOS", "Android (operating system)", "iOS", "JavaScript", "C++", 
    "Java (programming language)", "SQL", "Microprocessor", "Transistor", "Graphene", "Nanotechnology",
    
    # Cultural, Media & Medicine
    "David Bowie", "San Francisco", "Amoxicillin", "Sherlock (TV series)", "Pisces (astrology)",
    "The Beatles", "Harry Potter", "Star Wars", "Apple Inc.", "Google", "Microsoft"
]))

def fetch_wikipedia_content(title):
    """Fetch plain text of a Wikipedia page using the official MediaWiki Action API with retry backoff."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": 1,        # Only fetch the introduction summary paragraph(s) to optimize size
        "explaintext": 1,
        "titles": title,
        "format": "json",
        "redirects": 1
    }
    # Wikipedia API policy requires a unique, descriptive User-Agent with contact details
    headers = {
        "User-Agent": "LLM-Hallucination-Detector/1.0 (https://github.com/Nivetha15122006/hallucination-detector; nivetha15122006@gmail.com)"
    }
    
    max_retries = 5
    backoff_time = 2.0
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                pages = data.get("query", {}).get("pages", {})
                for page_id, page_data in pages.items():
                    if page_id == "-1":
                        print(f"[WARNING] Page not found on Wikipedia: {title}")
                        return None
                    return {
                        "title": page_data.get("title", title),
                        "text": page_data.get("extract", ""),
                        "url": f"https://en.wikipedia.org/wiki/{page_data.get('title', title).replace(' ', '_')}"
                    }
            elif response.status_code == 429:
                sleep_sec = backoff_time * (2 ** attempt)
                print(f"[WARNING] HTTP 429 Rate Limit for '{title}'. Retrying in {sleep_sec:.1f} seconds (attempt {attempt+1}/{max_retries})...")
                time.sleep(sleep_sec)
            else:
                print(f"[ERROR] Failed to fetch {title}: HTTP {response.status_code}")
                break
        except Exception as e:
            print(f"[ERROR] Exception fetching {title}: {e}")
            time.sleep(2)
            
    return None

def split_text(text, chunk_size=600, chunk_overlap=100):
    """Split text into overlapping chunks."""
    if USE_LANGCHAIN:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_text(text)
    else:
        # Fallback manual text splitter
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - chunk_overlap
        return chunks

def build_kb():
    print(f"Starting Optimized Knowledge Base Builder. Total topics defined: {len(TOPICS)}")
    
    # Define save paths relative to script location
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(data_dir, exist_ok=True)
    
    all_chunks = []
    fetched_count = 0
    
    for i, topic in enumerate(sorted(TOPICS)):
        print(f"[{i+1}/{len(TOPICS)}] Fetching '{topic}' summary...")
        content = fetch_wikipedia_content(topic)
        if content and len(content["text"].strip()) > 50:
            splits = split_text(content["text"], chunk_size=600, chunk_overlap=100)
            for split in splits:
                if len(split.strip()) > 30:
                    all_chunks.append({
                        "text": split.strip(),
                        "topic": content["title"],
                        "url": content["url"]
                    })
            fetched_count += 1
            # Lower sleep interval since summary responses are very small
            time.sleep(0.2)
        else:
            print(f"[WARNING] Skipping '{topic}' due to empty or missing content.")
            
    print(f"\nSuccessfully fetched {fetched_count}/{len(TOPICS)} summaries.")
    print(f"Created {len(all_chunks)} chunks total.")
    
    if not all_chunks:
        print("[ERROR] Error: No chunks created. Aborting.")
        return
        
    print("\nLoading sentence-transformers model 'all-MiniLM-L6-v2'...")
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generating embeddings for all chunks... (this will run very fast!)")
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = st_model.encode(texts, show_progress_bar=True, batch_size=64)
    
    print("\nBuilding FAISS FlatL2 index...")
    embeddings_np = np.array(embeddings, dtype=np.float32)
    dimension = embeddings_np.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    
    index_path = os.path.join(data_dir, "knowledge_base.index")
    chunks_path = os.path.join(data_dir, "chunks.pkl")
    
    print(f"Saving FAISS index to {index_path}...")
    faiss.write_index(index, index_path)
    
    print(f"Saving chunks metadata to {chunks_path}...")
    with open(chunks_path, "wb") as f:
        pickle.dump(all_chunks, f)
        
    print("\n[SUCCESS] Knowledge Base built successfully!")
    print(f"Index total vectors: {index.ntotal}")

if __name__ == "__main__":
    build_kb()
