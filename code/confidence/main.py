from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from confidence import calculate_confidence

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Confidence Service ready!")
    yield

app = FastAPI(
    title="Confidence Management Service",
    version="1.0.0",
    lifespan=lifespan
)

class ClassificationResult(BaseModel):
    predicted_class: str
    freshness:       str
    advice:          str
    probabilities:   dict

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "confidence_management"
    }

@app.post("/confidence")
def get_confidence(result: ClassificationResult):
    try:
        confidence = calculate_confidence(result.probabilities)
        return {
            "predicted_class": result.predicted_class,
            "freshness":       result.freshness,
            "advice":          result.advice,
            **confidence
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))