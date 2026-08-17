import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import requests

class HallucinationDetector:
    def __init__(self):
        # Read execution mode from environment variables
        self.use_hf_api = os.getenv("USE_HF_API", "false").lower() == "true"
        self.hf_token = os.getenv("HF_TOKEN", "")  # Optional: HF token for higher API limits
        
        self.labels = {0: 'FACTUAL', 1: 'UNCERTAIN', 2: 'HALLUCINATION'}
        self.tokenizer = None
        self.model = None
        
        # Load HuggingFace tokenizer (it's small and used to parse input lengths)
        self.model_name = "NiviG/hallucination-detector"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.use_hf_api:
            print("Detector initialized in Hugging Face Serverless API Mode! (Memory Optimized)")
        else:
            self._load()

    def _load(self):
        print("Loading model locally...")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model = self.model.float() # Ensure float32 for CPU LayerNorm compat
        self.model.eval()
        print("Model loaded locally!")

    def predict(self, claim: str, evidence: str):
        """
        Predict whether a claim is FACTUAL, UNCERTAIN, or a HALLUCINATION given an evidence paragraph.
        """
        if self.use_hf_api:
            # Query Hugging Face serverless Inference API (updated router endpoint)
            url = f"https://router.huggingface.co/hf-inference/models/NiviG/hallucination-detector"
            headers = {}
            if self.hf_token:
                headers["Authorization"] = f"Bearer {self.hf_token}"
                
            payload = {
                "inputs": {
                    "text": claim,
                    "text_pair": evidence
                },
                "options": {
                    "wait_for_model": True  # Force HF to spin up the model if sleeping
                }
            }
            
            import time
            for attempt in range(1, 4):
                try:
                    response = requests.post(url, json=payload, headers=headers, timeout=30)
                    
                    # If the model is sleeping, Hugging Face returns HTTP 503
                    if response.status_code == 503:
                        data = response.json()
                        estimated_time = data.get("estimated_time", 15)
                        print(f"⏳ HF Model is currently loading. Waiting {int(estimated_time)}s (Attempt {attempt}/3)...")
                        time.sleep(estimated_time)
                        continue
                        
                    if response.status_code == 200:
                        data = response.json()
                        predictions = data[0]
                        
                        scores = {"FACTUAL": 0.0, "UNCERTAIN": 0.0, "HALLUCINATION": 0.0}
                        for pred in predictions:
                            lbl = pred["label"]
                            score = float(pred["score"])
                            
                            # Map Hugging Face label output keys to our schema
                            if lbl == "LABEL_0" or lbl == "FACTUAL":
                                scores["FACTUAL"] = score
                            elif lbl == "LABEL_1" or lbl == "UNCERTAIN":
                                scores["UNCERTAIN"] = score
                            elif lbl == "LABEL_2" or lbl == "HALLUCINATION":
                                scores["HALLUCINATION"] = score
                                
                        pred_label = max(scores, key=scores.get)
                        raw_confidence = scores[pred_label]
                        
                        # Apply Calibration Logic
                        calibrated_label, calibrated_confidence = self._calibrate(pred_label, raw_confidence, scores)
                        
                        return {
                            'label': calibrated_label,
                            'confidence': calibrated_confidence,
                            'scores': scores
                        }
                    else:
                        raise RuntimeError(f"HF API returned HTTP {response.status_code}: {response.text}")
                except Exception as e:
                    print(f"[ERROR] HF Inference API attempt {attempt} failed: {e}")
                    if attempt == 3:
                        raise RuntimeError(f"Failed to query Hugging Face API after 3 attempts: {e}")
                    time.sleep(2)
        else:
            return self._predict_local(claim, evidence)

    def _predict_local(self, claim: str, evidence: str):
        """
        Run inference locally using PyTorch.
        """
        if self.model is None or self.tokenizer is None:
            self._load()
            
        # Swap input arguments to route correctly as (claim, evidence)
        inputs = self.tokenizer(
            claim,
            evidence,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
            
        # Map our fine-tuned DeBERTa model logits: LABEL_0 (Factual), LABEL_1 (Uncertain), LABEL_2 (Hallucination)
        scores = {
            "FACTUAL": float(probs[0]),
            "UNCERTAIN": float(probs[1]),
            "HALLUCINATION": float(probs[2])
        }
        
        pred_label = max(scores, key=scores.get)
        raw_confidence = scores[pred_label]
        
        # Apply Calibration Logic
        calibrated_label, calibrated_confidence = self._calibrate(pred_label, raw_confidence, scores)
        
        return {
            'label': calibrated_label,
            'confidence': calibrated_confidence,
            'scores': scores
        }

    def _calibrate(self, label: str, confidence: float, scores: dict):
        """
        Calibrate prediction confidence scores:
        - If FACTUAL/HALLUCINATION is < 70% confident, fall back to UNCERTAIN in the 50%-60% range.
        - If FACTUAL/HALLUCINATION is >= 70% confident, scale to 85%-99% range.
        - If UNCERTAIN, keep it in the 50s.
        """
        if label in ["FACTUAL", "HALLUCINATION"]:
            if confidence < 0.70:
                # Fall back to uncertain since we aren't completely sure
                new_label = "UNCERTAIN"
                # Map raw confidence (0.0 to 0.70) to 50%-60%
                new_confidence = 0.50 + (confidence / 0.70) * 0.10
                return new_label, new_confidence
            else:
                # Confident prediction: scale 70%-100% to 85%-99%
                new_confidence = 0.85 + ((confidence - 0.70) / (1.0 - 0.70)) * (0.99 - 0.85)
                return label, new_confidence
        else:
            # If model naturally predicted UNCERTAIN, ensure confidence is in the 50s (50%-59%)
            new_confidence = 0.50 + (confidence * 0.09)
            return label, new_confidence