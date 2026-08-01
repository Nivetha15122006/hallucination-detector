import os
import sys

# Configure HuggingFace cache directory to local folder to avoid permission errors
os.environ["HF_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hf_cache"))
os.environ["TRANSFORMERS_CACHE"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hf_cache"))

# Import retriever and model from the local backend directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from retriever import Retriever
from model import HallucinationDetector

def main():
    print("==================================================")
    print("🔍 LLM Hallucination Detector — Pipeline Test 🔍")
    print("==================================================")
    
    print("\n1. Initializing Retriever (loading FAISS index)...")
    try:
        retriever = Retriever()
    except Exception as e:
        print(f"❌ Error loading retriever: {e}")
        print("Please make sure you run 'python backend/build_knowledge_base.py' first.")
        return
        
    print("\n2. Initializing Detector (loading DeBERTa-v3 from cache)...")
    detector = HallucinationDetector()
    
    print("\n3. Executing Test Cases...")
    
    # Define test questions and claims
    test_cases = [
        {
            "question": "Who invented the telephone?",
            "claims": [
                ("Factual Claim", "Alexander Graham Bell invented the telephone."),
                ("Hallucinated Claim", "Albert Einstein invented the telephone.")
            ]
        },
        {
            "question": "David Bowie starred in what?",
            "claims": [
                ("Factual Claim", "David Bowie starred in the cult film The Man Who Fell to Earth."),
                ("Hallucinated Claim", "David Bowie was not an actor and never starred in any movies.")
            ]
        }
    ]
    
    for tc in test_cases:
        question = tc["question"]
        print("\n--------------------------------------------------")
        print(f"❓ Question: '{question}'")
        
        # Retrieve evidence
        evidence_list = retriever.retrieve(question, top_k=1)
        if not evidence_list:
            print("❌ No evidence found in RAG knowledge base.")
            continue
            
        evidence = evidence_list[0]["text"]
        topic = evidence_list[0]["topic"]
        print(f"📄 Retrieved Wikipedia Context (Topic: '{topic}'):")
        print(f"   \"{evidence}\"")
        
        for claim_type, claim in tc["claims"]:
            print(f"\n👉 Testing {claim_type}: \"{claim}\"")
            result = detector.predict(claim=claim, evidence=evidence)
            
            # Map predictions
            label = result['label']
            confidence = result['confidence'] * 100
            
            emoji = "✅" if label == "FACTUAL" else "⚠️" if label == "HALLUCINATION" else "❓"
            print(f"   Result: [{emoji} {label}] with {confidence:.1f}% confidence")
            print(f"   Probabilities: FACTUAL={result['scores']['FACTUAL']:.4f}, UNCERTAIN={result['scores']['UNCERTAIN']:.4f}, HALLUCINATION={result['scores']['HALLUCINATION']:.4f}")

if __name__ == "__main__":
    main()
