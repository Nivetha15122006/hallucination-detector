import requests
import json

def main():
    url = "http://localhost:8000/check"
    
    print("==================================================")
    print("🌐 LLM Hallucination Detector — API Endpoint Test 🌐")
    print("==================================================")
    print("This script queries a running FastAPI local server.")
    print("Ensure you ran: python backend/main.py in another terminal.")
    print("--------------------------------------------------")
    
    test_cases = [
        {
            "question": "Who invented the telephone?",
            "ai_answer": "Alexander Graham Bell invented the telephone."
        },
        {
            "question": "Who invented the telephone?",
            "ai_answer": "Albert Einstein invented the telephone."
        },
        {
            "question": "Where was Alexander Graham Bell born?",
            "ai_answer": "Alexander Graham Bell was born in Berlin, Germany."
        }
    ]
    
    headers = {"Content-Type": "application/json"}
    
    for i, payload in enumerate(test_cases):
        print(f"\n[Test Case {i+1}] Sending Check Query...")
        print(f"Question:  \"{payload['question']}\"")
        print(f"AI Answer: \"{payload['ai_answer']}\"")
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            if response.status_code == 200:
                data = response.json()
                print(f"Result Label: {data['label']}")
                print(f"Summary:      {data['summary']}")
                print("Retrieved Evidence Source:")
                print(f"   Topic: {data['evidence'][0]['topic']}")
                print(f"   URL:   {data['evidence'][0]['url']}")
                print(f"   Text:  \"{data['evidence'][0]['text']}\"")
            else:
                print(f"❌ Error: API returned HTTP {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("Please ensure the backend is running at http://localhost:8000")
            break

if __name__ == "__main__":
    main()
