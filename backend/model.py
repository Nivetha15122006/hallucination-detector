import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class HallucinationDetector:
    def __init__(self):
        self.device = torch.device('cpu')
        self.tokenizer = None
        self.model = None
        self.labels = {0: 'FACTUAL', 1: 'UNCERTAIN', 2: 'HALLUCINATION'}
        self._load()

    def _load(self):
        print("Loading model...")
        model_name = "NiviG/hallucination-detector"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load in float32 to avoid LayerNorm errors on CPU
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        self.model = self.model.float() # Ensure float32
        self.model.eval()
        print("Model loaded!")

    def predict(self, claim: str, evidence: str):
        """
        Predict whether a claim is FACTUAL, UNCERTAIN, or a HALLUCINATION given an evidence paragraph.
        
        The model was trained with:
          Text 1 (First argument): The claim to check
          Text 2 (Second argument): The evidence context
        """
        inputs = self.tokenizer(
            claim,      # Claim is the first sentence
            evidence,   # Evidence is the second sentence
            truncation=True,
            max_length=256, # Matches training max_length
            padding='max_length',
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits.float(), dim=1).numpy()[0]
            pred_id = int(np.argmax(probs))

        return {
            'label': self.labels[pred_id],
            'confidence': float(probs[pred_id]),
            'scores': {
                'FACTUAL': float(probs[0]),
                'UNCERTAIN': float(probs[1]),
                'HALLUCINATION': float(probs[2])
            }
        }