from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from retriever import Retriever
from model import HallucinationDetector
import uvicorn

app = FastAPI(
    title="LLM Hallucination Detector API",
    description="Detects hallucinations in LLM outputs using DeBERTa + RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

print("Loading models...")
retriever = Retriever()
detector = HallucinationDetector()
print("All models loaded! API ready!")

class CheckRequest(BaseModel):
    question: str
    ai_answer: str

class CheckResponse(BaseModel):
    label: str
    confidence: float
    scores: dict
    evidence: list
    summary: str

@app.get("/")
def root():
    return {"message": "LLM Hallucination Detector API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/check", response_model=CheckResponse)
def check_hallucination(request: CheckRequest):
    try:
        evidence = retriever.retrieve(request.question, top_k=3)

        if not evidence:
            raise HTTPException(status_code=404, detail="No evidence found")

        top_evidence = evidence[0]['text']
        result = detector.predict(
            premise=top_evidence,
            hypothesis=request.ai_answer
        )

        label = result['label']
        confidence = round(result['confidence'] * 100, 1)

        if label == 'FACTUAL':
            summary = f"✅ This answer appears factual ({confidence}% confidence)"
        elif label == 'HALLUCINATION':
            summary = f"⚠️ This answer may be incorrect ({confidence}% confidence)"
        else:
            summary = f"❓ Cannot verify this answer ({confidence}% confidence)"

        return CheckResponse(
            label=label,
            confidence=result['confidence'],
            scores=result['scores'],
            evidence=evidence,
            summary=summary
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)