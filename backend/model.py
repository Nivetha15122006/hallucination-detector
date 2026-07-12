import requests
import os

class HallucinationDetector:
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/NiviG/hallucination-detector"
        self.headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN', '')}"}
        self.labels = {0: 'FACTUAL', 1: 'UNCERTAIN', 2: 'HALLUCINATION'}
        print("HuggingFace Inference API ready!")

    def predict(self, premise: str, hypothesis: str):
        try:
            payload = {
                "inputs": {
                    "text": premise,
                    "text_pair": hypothesis
                }
            }
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                # Parse HuggingFace response
                if isinstance(result, list) and len(result) > 0:
                    scores = {item['label']: item['score'] for item in result[0]}
                    
                    # Map labels
                    label_map = {
                        'LABEL_0': 'FACTUAL',
                        'LABEL_1': 'UNCERTAIN', 
                        'LABEL_2': 'HALLUCINATION'
                    }
                    
                    best_label = max(scores, key=scores.get)
                    mapped_label = label_map.get(best_label, 'UNCERTAIN')
                    confidence = scores[best_label]
                    
                    return {
                        'label': mapped_label,
                        'confidence': float(confidence),
                        'scores': {
                            'FACTUAL': float(scores.get('LABEL_0', 0)),
                            'UNCERTAIN': float(scores.get('LABEL_1', 0)),
                            'HALLUCINATION': float(scores.get('LABEL_2', 0))
                        }
                    }
        except Exception as e:
            print(f"HF API error: {e}")
        
        # Fallback
        return {
            'label': 'UNCERTAIN',
            'confidence': 0.5,
            'scores': {'FACTUAL': 0.33, 'UNCERTAIN': 0.34, 'HALLUCINATION': 0.33}
        }