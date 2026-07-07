import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class HallucinationDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.labels = {0: 'FACTUAL', 1: 'UNCERTAIN', 2: 'HALLUCINATION'}
        self._load()

    def _load(self):
        print("Loading DeBERTa model...")
        model_name = "NiviG/hallucination-detector"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        self.model = self.model.float()
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}!")

    def predict(self, premise: str, hypothesis: str):
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=256,
            padding='max_length',
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_id = np.argmax(probs)

        return {
            'label': self.labels[pred_id],
            'confidence': float(probs[pred_id]),
            'scores': {
                'FACTUAL': float(probs[0]),
                'UNCERTAIN': float(probs[1]),
                'HALLUCINATION': float(probs[2])
            }
        }