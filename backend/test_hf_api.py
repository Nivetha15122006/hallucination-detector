import requests
import json
import time

def test_hf_api():
    model_id = "NiviG/hallucination-detector"
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    # Test payload
    payload = {
        "inputs": {
            "text": "Alexander Graham Bell invented the telephone.",
            "text_pair": "Alexander Graham Bell was a Scottish-American inventor, scientist, and engineer who is credited with patenting the first practical telephone."
        }
    }
    
    # We will try up to 3 times in case the model is waking up (lazy loading on HuggingFace servers)
    for attempt in range(1, 4):
        print(f"\n[Attempt {attempt}] Querying Hugging Face Serverless API for {model_id}...")
        try:
            response = requests.post(url, json=payload, timeout=20)
            
            # If the model is sleeping, HuggingFace will return HTTP 503 while it loads the model
            if response.status_code == 503:
                data = response.json()
                estimated_time = data.get("estimated_time", 20)
                print(f"⏳ Model is currently loading on Hugging Face. Waiting {int(estimated_time)} seconds to wake it up...")
                time.sleep(estimated_time)
                continue
                
            if response.status_code == 200:
                print("✅ API Responded Successfully!")
                print("Response JSON:")
                print(json.dumps(response.json(), indent=2))
                return
            else:
                print(f"❌ API returned HTTP {response.status_code}:")
                print(response.text)
                return
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return
            
    print("\n❌ Failed to get a response after waking up the model.")

if __name__ == "__main__":
    test_hf_api()
