import requests
import os

class HallucinationDetector:
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/NiviG/hallucination-detector"
        self.headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN', '')}"}
        self.id2label = {0: 'FACTUAL', 1: 'UNCERTAIN', 2: 'HALLUCINATION'}
        print("HuggingFace Inference API ready!")

    def predict(self, premise: str, hypothesis: str):
        try:
            # Send as NLI pair
            payload = {
                "inputs": f"{premise} [SEP] {hypothesis}"
            }
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            print(f"HF API status: {response.status_code}")
            print(f"HF API response: {response.text[:200]}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Result type: {type(result)}")
                
                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        items = result[0]
                    else:
                        items = result
                    
                    # Get scores
                    scores = {}
                    for item in items:
                        if isinstance(item, dict) and 'label' in item:
                            scores[item['label']] = item['score']
                    
                    print(f"Scores: {scores}")
                    
                    if scores:
                        best_label = max(scores, key=scores.get)
                        confidence = scores[best_label]
                        
                        # Map LABEL_0, LABEL_1, LABEL_2
                        label_map = {
                            'LABEL_0': 'FACTUAL',
                            'LABEL_1': 'UNCERTAIN',
                            'LABEL_2': 'HALLUCINATION'
                        }
                        mapped = label_map.get(best_label, best_label)
                        
                        # Also try direct mapping
                        if best_label in ['FACTUAL', 'UNCERTAIN', 'HALLUCINATION']:
                            mapped = best_label
                        
                        return {
                            'label': mapped,
                            'confidence': float(confidence),
                            'scores': {
                                'FACTUAL': float(scores.get('LABEL_0', scores.get('FACTUAL', 0.33))),
                                'UNCERTAIN': float(scores.get('LABEL_1', scores.get('UNCERTAIN', 0.34))),
                                'HALLUCINATION': float(scores.get('LABEL_2', scores.get('HALLUCINATION', 0.33)))
                            }
                        }

        except Exception as e:
            print(f"HF API error: {e}")
        
        return {
            'label': 'UNCERTAIN',
            'confidence': 0.5,
            'scores': {'FACTUAL': 0.33, 'UNCERTAIN': 0.34, 'HALLUCINATION': 0.33}
        }