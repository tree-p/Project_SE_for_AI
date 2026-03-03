from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from torchvision import models
from model import load_model, classify_from_base64, classify_from_bytes
import torch
import os

WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "weights/beef_model.pt")

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not os.path.exists(WEIGHTS_PATH):
        print("WARNING: weights not found — using pretrained for dev")
        m = models.efficientnet_b0(weights="DEFAULT")
        m.classifier[1] = torch.nn.Linear(1280, 3)
        model = m
        model.eval()
    else:
        model = load_model(WEIGHTS_PATH)
    print("Model loaded successfully")
    yield

app = FastAPI(
    title="Freshness Classification Service",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "freshness_classification"
    }

@app.post("/classify/upload")
async def classify_upload(file: UploadFile = File(...)):
    """รับภาพโดยตรง"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    image_bytes = await file.read()
    return classify_from_bytes(model, image_bytes)

class ROIRequest(BaseModel):
    roi_base64: str

@app.post("/classify/roi")
def classify_roi(body: ROIRequest):
    try:
        return classify_from_base64(model, body.roi_base64)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))