from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from beef_localization import localize_beef
from classifier import load_model, classify_from_base64
from confidence import calculate_confidence

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Beef Freshness API running"}

@app.options("/analyze")
async def options_analyze():
    return JSONResponse(content={"message": "OK"})


# โหลดโมเดลตอน start server
model = load_model("models/beef_model.pt")


@app.post("/analyze")
async def analyze_beef(file: UploadFile = File(...)):

    image_bytes = await file.read()

    roi_result = localize_beef(image_bytes)

    if roi_result.get("status") == "rejected":
        return roi_result

    roi_b64 = roi_result["roi_base64"]

    result = classify_from_base64(model, roi_b64)

    confidence = calculate_confidence(result["probabilities"])

    return {
        "prediction": result["predicted_class"],
        "freshness": result["freshness"],
        "advice": result["advice"],
        "probabilities": result["probabilities"],
        "confidence": confidence,
        "roi_bbox": roi_result["bbox"],
        "localization_confidence": roi_result["localization_confidence"],
        "fallback": roi_result["fallback"]
    }
