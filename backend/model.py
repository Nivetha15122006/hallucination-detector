import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os

class HallucinationDetector:
    def __init__(self):
        self.device = torch.device('cpu')
        self.tokenizer = None
        self.model = None
        self.labels = {0: 'FACTUAL', 1: 'UNCERTAIN', 2: 'HALLUCINATION'}
        self._load()

    def _load(self):
        print("Loading DeBERTa model...")
        model_name = "NiviG/hallucination-detector"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            low_cpu_mem_usage=True
        )
        self.model = self.model.float()
        self.model.eval()
        print("Model loaded!")

    def predict(self, premise: str, hypothesis: str):
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=128,
            padding='max_length',
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
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