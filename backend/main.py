from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="LLM Hallucination Detector API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Lazy loading — models load on first request not startup
retriever = None
detector = None

def get_retriever():
    global retriever
    if retriever is None:
        from retriever import Retriever
        retriever = Retriever()
    return retriever

def get_detector():
    global detector
    if detector is None:
        from model import HallucinationDetector
        detector = HallucinationDetector()
    return detector

class CheckRequest(BaseModel):
    question: str
    ai_answer: str

@app.get("/")
def root():
    return {"message": "Hallucination Detector API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/check")
def check_hallucination(request: CheckRequest):
    try:
        r = get_retriever()
        d = get_detector()

        evidence = r.retrieve(request.question, top_k=3)
        if not evidence:
            raise HTTPException(status_code=404, detail="No evidence found")

        top_evidence = evidence[0]['text']
        result = d.predict(
            premise=top_evidence,
            hypothesis=request.ai_answer
        )

        label = result['label']
        confidence = round(result['confidence'] * 100, 1)

        if label == 'FACTUAL':
            summary = f"✅ Appears factual ({confidence}% confidence)"
        elif label == 'HALLUCINATION':
            summary = f"⚠️ May be incorrect ({confidence}% confidence)"
        else:
            summary = f"❓ Cannot verify ({confidence}% confidence)"

        return {
            'label': label,
            'confidence': result['confidence'],
            'scores': result['scores'],
            'evidence': evidence,
            'summary': summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)